"""
Business Rules Engine per Forecasting

Gestisce regole business e vincoli per forecast interpretabili.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Callable, Tuple
from dataclasses import dataclass
from enum import Enum
from pydantic import BaseModel, Field
import json
import numpy as np
import pandas as pd

from arima_forecaster.utils.logger import get_logger

logger = get_logger(__name__)


class RuleType(Enum):
    """Tipi di regole business"""
    CONSTRAINT = "constraint"           # Vincolo hard (es. valore massimo)
    ADJUSTMENT = "adjustment"           # Aggiustamento valore (es. cap, floor)
    VALIDATION = "validation"           # Validazione logica (es. coerenza)
    ALERT = "alert"                    # Generazione alert (es. soglie)
    TRANSFORMATION = "transformation"   # Trasformazione valore (es. round)
    CONDITIONAL = "conditional"         # Regola condizionale (if-then)


class RuleAction(Enum):
    """Azioni per regole business"""
    ACCEPT = "accept"                  # Accetta valore come è
    REJECT = "reject"                  # Rifiuta valore
    MODIFY = "modify"                  # Modifica valore
    ALERT_ONLY = "alert_only"          # Solo alert, non modifica
    OVERRIDE = "override"              # Override completo
    ESCALATE = "escalate"              # Escalation a supervisore


class Rule(BaseModel):
    """Schema regola business"""
    id: str = Field(..., description="ID univoco regola")
    name: str = Field(..., description="Nome descrittivo")
    description: str = Field(..., description="Descrizione dettagliata")
    rule_type: RuleType = Field(..., description="Tipo regola")
    condition: str = Field(..., description="Condizione Python da valutare")
    action: RuleAction = Field(..., description="Azione da eseguire")
    priority: int = Field(default=1, description="Priorità (1=alta, 10=bassa)")
    enabled: bool = Field(default=True, description="Regola attiva")
    
    # Parametri azione
    action_params: Dict[str, Any] = Field(default_factory=dict)
    
    # Metadati
    created_by: str = Field(default="system")
    created_at: datetime = Field(default_factory=datetime.now)
    last_applied: Optional[datetime] = None
    application_count: int = Field(default=0)
    
    # Validazione e logging
    log_applications: bool = Field(default=True)
    require_approval: bool = Field(default=False)
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class RuleResult(BaseModel):
    """Risultato applicazione regola"""
    rule_id: str
    rule_name: str
    applied: bool
    action_taken: RuleAction
    original_value: float
    modified_value: Optional[float] = None
    message: str
    confidence: float = Field(default=1.0)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.now)


class BusinessContext(BaseModel):
    """Contesto business per valutazione regole"""
    model_id: str
    product_id: Optional[str] = None
    category: Optional[str] = None
    channel: Optional[str] = None
    region: Optional[str] = None
    customer_segment: Optional[str] = None
    
    # Dati temporali
    forecast_date: datetime
    is_weekend: bool = False
    is_holiday: bool = False
    season: Optional[str] = None
    
    # Metriche business
    historical_average: Optional[float] = None
    trend: Optional[float] = None
    seasonality_factor: Optional[float] = None
    
    # Vincoli operativi
    min_capacity: Optional[float] = None
    max_capacity: Optional[float] = None
    budget_limit: Optional[float] = None
    
    # Dati esterni
    weather_forecast: Optional[Dict[str, Any]] = None
    marketing_events: Optional[List[Dict[str, Any]]] = None
    competitor_actions: Optional[List[Dict[str, Any]]] = None
    
    # Metadati aggiuntivi
    custom_attributes: Dict[str, Any] = Field(default_factory=dict)


@dataclass
class RuleEngineConfig:
    """Configurazione engine regole business"""
    max_rules_per_context: int = 50
    rule_timeout_seconds: float = 5.0
    enable_rule_chaining: bool = True
    log_all_evaluations: bool = False
    default_confidence_threshold: float = 0.8
    enable_approval_workflow: bool = False


class BusinessRulesEngine:
    """
    Engine per gestione regole business nel forecasting
    
    Features:
    - Valutazione regole condizionali
    - Prioritizzazione e ordinamento
    - Modifiche valori forecast
    - Logging e audit trail
    - Workflow approvazione
    - Spiegazioni decisioni
    """
    
    def __init__(self, config: RuleEngineConfig = None):
        self.config = config or RuleEngineConfig()
        self.rules: Dict[str, Rule] = {}
        self.rule_history: List[RuleResult] = []
        self.approval_queue: List[Dict[str, Any]] = []
        
        # Statistiche
        self.evaluations_count = 0
        self.modifications_count = 0
        self.alerts_count = 0
        
        # Inizializza regole predefinite
        self._initialize_default_rules()
    
    def _initialize_default_rules(self):
        """Inizializza set di regole business comuni"""
        
        default_rules = [
            # Vincoli di capacità
            Rule(
                id="capacity_max_limit",
                name="Limite Massimo Capacità",
                description="Limita forecast al massimo della capacità produttiva",
                rule_type=RuleType.CONSTRAINT,
                condition="forecast_value > context.max_capacity if context.max_capacity else False",
                action=RuleAction.MODIFY,
                action_params={"new_value": "context.max_capacity", "message": "Forecast limitato alla capacità massima"},
                priority=1
            ),
            
            Rule(
                id="capacity_min_limit", 
                name="Limite Minimo Operativo",
                description="Impone forecast minimo per mantenere operazioni base",
                rule_type=RuleType.CONSTRAINT,
                condition="forecast_value < context.min_capacity if context.min_capacity else False",
                action=RuleAction.MODIFY,
                action_params={"new_value": "context.min_capacity", "message": "Forecast alzato al minimo operativo"},
                priority=2
            ),
            
            # Validazioni di coerenza
            Rule(
                id="negative_forecast_check",
                name="Controllo Forecast Negativi",
                description="Blocca forecast negativi quando non logici",
                rule_type=RuleType.VALIDATION,
                condition="forecast_value < 0 and context.category not in ['returns', 'adjustments']",
                action=RuleAction.MODIFY,
                action_params={"new_value": "0", "message": "Forecast negativo corretto a zero"},
                priority=1
            ),
            
            # Alert per anomalie
            Rule(
                id="significant_increase_alert",
                name="Alert Aumento Significativo",
                description="Alert per aumenti >50% rispetto alla media storica",
                rule_type=RuleType.ALERT,
                condition="forecast_value > (context.historical_average * 1.5) if context.historical_average else False",
                action=RuleAction.ALERT_ONLY,
                action_params={"message": "Forecast significativamente superiore alla media storica"},
                priority=3
            ),
            
            Rule(
                id="weekend_adjustment",
                name="Aggiustamento Weekend", 
                description="Riduce forecast nei weekend per categorie B2B",
                rule_type=RuleType.ADJUSTMENT,
                condition="context.is_weekend and context.category == 'B2B'",
                action=RuleAction.MODIFY,
                action_params={"multiplier": "0.3", "message": "Forecast ridotto per weekend B2B"},
                priority=4
            ),
            
            # Vincoli budget
            Rule(
                id="budget_constraint",
                name="Vincolo Budget",
                description="Limita forecast in base al budget disponibile",
                rule_type=RuleType.CONSTRAINT,
                condition="forecast_value > context.budget_limit if context.budget_limit else False",
                action=RuleAction.MODIFY,
                action_params={"new_value": "context.budget_limit", "message": "Forecast limitato dal budget disponibile"},
                priority=2
            ),
            
            # Arrotondamenti business
            Rule(
                id="round_to_units",
                name="Arrotondamento Unità",
                description="Arrotonda forecast a unità intere per prodotti discreti",
                rule_type=RuleType.TRANSFORMATION,
                condition="context.category in ['units', 'pieces', 'items'] and forecast_value != int(forecast_value)",
                action=RuleAction.MODIFY,
                action_params={"transform": "round", "message": "Forecast arrotondato a unità intere"},
                priority=5
            )
        ]
        
        # Registra regole
        for rule in default_rules:
            self.add_rule(rule)
        
        logger.info(f"Inizializzate {len(default_rules)} regole business predefinite")
    
    def add_rule(self, rule: Rule) -> bool:
        """Aggiunge regola al engine"""
        try:
            # Valida regola
            if not self._validate_rule(rule):
                logger.error(f"Regola non valida: {rule.id}")
                return False
            
            self.rules[rule.id] = rule
            logger.info(f"Regola aggiunta: {rule.name} ({rule.id})")
            return True
            
        except Exception as e:
            logger.error(f"Errore aggiunta regola {rule.id}: {e}")
            return False
    
    def remove_rule(self, rule_id: str) -> bool:
        """Rimuove regola dal engine"""
        if rule_id in self.rules:
            removed_rule = self.rules.pop(rule_id)
            logger.info(f"Regola rimossa: {removed_rule.name}")
            return True
        return False
    
    def enable_rule(self, rule_id: str, enabled: bool = True) -> bool:
        """Abilita/disabilita regola"""
        if rule_id in self.rules:
            self.rules[rule_id].enabled = enabled
            status = "abilitata" if enabled else "disabilitata"
            logger.info(f"Regola {rule_id} {status}")
            return True
        return False
    
    def _validate_rule(self, rule: Rule) -> bool:
        """Valida sintassi e logica regola"""
        try:
            # Controlla ID univoco
            if rule.id in self.rules:
                logger.warning(f"Regola {rule.id} già esistente, verrà sostituita")
            
            # Valida condizione Python (controllo sintattico)
            try:
                compile(rule.condition, '<rule_condition>', 'eval')
            except SyntaxError as e:
                logger.error(f"Errore sintassi condizione regola {rule.id}: {e}")
                return False
            
            # Valida parametri azione
            if rule.action == RuleAction.MODIFY:
                required_params = ["new_value", "multiplier", "transform"]
                if not any(param in rule.action_params for param in required_params):
                    logger.error(f"Regola {rule.id}: azione MODIFY richiede parametri di modifica")
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Errore validazione regola {rule.id}: {e}")
            return False
    
    def apply_rules(
        self, 
        forecast_value: float,
        context: BusinessContext,
        rule_ids: Optional[List[str]] = None
    ) -> Tuple[float, List[RuleResult]]:
        """
        Applica regole business a valore forecast
        
        Args:
            forecast_value: Valore originale del forecast
            context: Contesto business
            rule_ids: Lista specifica regole da applicare (opzionale)
            
        Returns:
            (valore_modificato, lista_risultati_regole)
        """
        try:
            self.evaluations_count += 1
            start_time = datetime.now()
            
            # Seleziona regole da applicare
            applicable_rules = self._get_applicable_rules(rule_ids)
            
            # Ordina per priorità
            applicable_rules.sort(key=lambda r: r.priority)
            
            current_value = forecast_value
            results = []
            
            logger.debug(f"Applicazione {len(applicable_rules)} regole a valore {forecast_value}")
            
            # Applica regole in sequenza
            for rule in applicable_rules:
                try:
                    result = self._apply_single_rule(rule, current_value, context)
                    results.append(result)
                    
                    # Aggiorna valore se modificato
                    if result.applied and result.modified_value is not None:
                        current_value = result.modified_value
                        self.modifications_count += 1
                    
                    # Conta alert
                    if result.action_taken == RuleAction.ALERT_ONLY:
                        self.alerts_count += 1
                    
                    # Aggiorna statistiche regola
                    rule.application_count += 1
                    rule.last_applied = datetime.now()
                    
                    # Logging se abilitato
                    if self.config.log_all_evaluations or result.applied:
                        logger.info(f"Regola {rule.name}: {result.message}")
                
                except Exception as e:
                    logger.error(f"Errore applicazione regola {rule.id}: {e}")
                    continue
            
            # Salva in history
            self.rule_history.extend(results)
            
            # Limita history size
            if len(self.rule_history) > 1000:
                self.rule_history = self.rule_history[-800:]
            
            elapsed = (datetime.now() - start_time).total_seconds()
            logger.debug(f"Regole applicate in {elapsed:.3f}s: {forecast_value} → {current_value}")
            
            return current_value, results
            
        except Exception as e:
            logger.error(f"Errore applicazione regole: {e}")
            return forecast_value, []
    
    def _get_applicable_rules(self, rule_ids: Optional[List[str]] = None) -> List[Rule]:
        """Ottiene lista regole applicabili"""
        if rule_ids:
            # Regole specifiche richieste
            return [self.rules[rid] for rid in rule_ids if rid in self.rules and self.rules[rid].enabled]
        else:
            # Tutte le regole abilitate
            return [rule for rule in self.rules.values() if rule.enabled]
    
    def _apply_single_rule(
        self, 
        rule: Rule, 
        current_value: float, 
        context: BusinessContext
    ) -> RuleResult:
        """Applica singola regola"""
        try:
            # Prepara contesto per valutazione
            eval_context = {
                'forecast_value': current_value,
                'context': context,
                'rule': rule,
                'abs': abs,
                'min': min,
                'max': max,
                'round': round,
                'int': int,
                'float': float,
                'datetime': datetime,
                'timedelta': timedelta
            }
            
            # Valuta condizione
            try:
                condition_result = eval(rule.condition, {"__builtins__": {}}, eval_context)
            except Exception as e:
                logger.error(f"Errore valutazione condizione regola {rule.id}: {e}")
                return RuleResult(
                    rule_id=rule.id,
                    rule_name=rule.name,
                    applied=False,
                    action_taken=RuleAction.REJECT,
                    original_value=current_value,
                    message=f"Errore valutazione condizione: {e}"
                )
            
            # Se condizione non soddisfatta, non applicare regola
            if not condition_result:
                return RuleResult(
                    rule_id=rule.id,
                    rule_name=rule.name,
                    applied=False,
                    action_taken=RuleAction.ACCEPT,
                    original_value=current_value,
                    message="Condizione non soddisfatta"
                )
            
            # Applica azione
            return self._execute_rule_action(rule, current_value, context, eval_context)
            
        except Exception as e:
            logger.error(f"Errore applicazione regola {rule.id}: {e}")
            return RuleResult(
                rule_id=rule.id,
                rule_name=rule.name,
                applied=False,
                action_taken=RuleAction.REJECT,
                original_value=current_value,
                message=f"Errore interno: {e}"
            )
    
    def _execute_rule_action(
        self,
        rule: Rule,
        current_value: float,
        context: BusinessContext,
        eval_context: Dict[str, Any]
    ) -> RuleResult:
        """Esegue azione della regola"""
        
        action_params = rule.action_params
        message = action_params.get("message", f"Regola {rule.name} applicata")
        
        try:
            if rule.action == RuleAction.ACCEPT:
                return RuleResult(
                    rule_id=rule.id,
                    rule_name=rule.name,
                    applied=True,
                    action_taken=rule.action,
                    original_value=current_value,
                    message=message
                )
            
            elif rule.action == RuleAction.REJECT:
                return RuleResult(
                    rule_id=rule.id,
                    rule_name=rule.name,
                    applied=True,
                    action_taken=rule.action,
                    original_value=current_value,
                    message=f"Valore rifiutato: {message}"
                )
            
            elif rule.action == RuleAction.MODIFY:
                # Calcola nuovo valore
                new_value = self._calculate_modified_value(current_value, action_params, eval_context)
                
                return RuleResult(
                    rule_id=rule.id,
                    rule_name=rule.name,
                    applied=True,
                    action_taken=rule.action,
                    original_value=current_value,
                    modified_value=new_value,
                    message=f"{message} ({current_value:.2f} → {new_value:.2f})"
                )
            
            elif rule.action == RuleAction.ALERT_ONLY:
                return RuleResult(
                    rule_id=rule.id,
                    rule_name=rule.name,
                    applied=True,
                    action_taken=rule.action,
                    original_value=current_value,
                    message=f"ALERT: {message}"
                )
            
            elif rule.action == RuleAction.OVERRIDE:
                override_value = action_params.get("override_value", current_value)
                if isinstance(override_value, str):
                    override_value = eval(override_value, {"__builtins__": {}}, eval_context)
                
                return RuleResult(
                    rule_id=rule.id,
                    rule_name=rule.name,
                    applied=True,
                    action_taken=rule.action,
                    original_value=current_value,
                    modified_value=float(override_value),
                    message=f"Override applicato: {message}"
                )
            
            elif rule.action == RuleAction.ESCALATE:
                # Aggiunge a coda approvazione se abilitata
                if self.config.enable_approval_workflow:
                    self.approval_queue.append({
                        "rule_id": rule.id,
                        "original_value": current_value,
                        "context": context.dict(),
                        "timestamp": datetime.now(),
                        "message": message
                    })
                
                return RuleResult(
                    rule_id=rule.id,
                    rule_name=rule.name,
                    applied=True,
                    action_taken=rule.action,
                    original_value=current_value,
                    message=f"Escalation richiesta: {message}"
                )
            
            else:
                return RuleResult(
                    rule_id=rule.id,
                    rule_name=rule.name,
                    applied=False,
                    action_taken=RuleAction.ACCEPT,
                    original_value=current_value,
                    message=f"Azione non riconosciuta: {rule.action}"
                )
            
        except Exception as e:
            logger.error(f"Errore esecuzione azione regola {rule.id}: {e}")
            return RuleResult(
                rule_id=rule.id,
                rule_name=rule.name,
                applied=False,
                action_taken=RuleAction.REJECT,
                original_value=current_value,
                message=f"Errore esecuzione: {e}"
            )
    
    def _calculate_modified_value(
        self,
        current_value: float,
        action_params: Dict[str, Any],
        eval_context: Dict[str, Any]
    ) -> float:
        """Calcola valore modificato basandosi sui parametri"""
        
        # Nuovo valore esplicito
        if "new_value" in action_params:
            new_val = action_params["new_value"]
            if isinstance(new_val, str):
                return float(eval(new_val, {"__builtins__": {}}, eval_context))
            return float(new_val)
        
        # Moltiplicatore
        elif "multiplier" in action_params:
            multiplier = action_params["multiplier"]
            if isinstance(multiplier, str):
                multiplier = eval(multiplier, {"__builtins__": {}}, eval_context)
            return current_value * float(multiplier)
        
        # Aggiustamento additivo
        elif "adjustment" in action_params:
            adjustment = action_params["adjustment"]
            if isinstance(adjustment, str):
                adjustment = eval(adjustment, {"__builtins__": {}}, eval_context)
            return current_value + float(adjustment)
        
        # Trasformazione
        elif "transform" in action_params:
            transform = action_params["transform"]
            if transform == "round":
                return round(current_value)
            elif transform == "floor":
                return float(int(current_value))
            elif transform == "ceil":
                return float(int(current_value) + (1 if current_value % 1 > 0 else 0))
            elif transform == "abs":
                return abs(current_value)
        
        # Cap (massimo)
        elif "cap" in action_params:
            cap = action_params["cap"]
            if isinstance(cap, str):
                cap = eval(cap, {"__builtins__": {}}, eval_context)
            return min(current_value, float(cap))
        
        # Floor (minimo)
        elif "floor" in action_params:
            floor_val = action_params["floor"]
            if isinstance(floor_val, str):
                floor_val = eval(floor_val, {"__builtins__": {}}, eval_context)
            return max(current_value, float(floor_val))
        
        # Default: nessuna modifica
        return current_value
    
    def get_rule_explanations(
        self, 
        forecast_value: float, 
        context: BusinessContext,
        results: List[RuleResult]
    ) -> Dict[str, Any]:
        """Genera spiegazioni per applicazione regole"""
        try:
            applied_rules = [r for r in results if r.applied]
            modifications = [r for r in applied_rules if r.modified_value is not None]
            alerts = [r for r in applied_rules if r.action_taken == RuleAction.ALERT_ONLY]
            
            explanation = {
                "original_value": forecast_value,
                "final_value": modifications[-1].modified_value if modifications else forecast_value,
                "total_rules_evaluated": len(results),
                "rules_applied": len(applied_rules),
                "modifications_made": len(modifications),
                "alerts_generated": len(alerts),
                "processing_summary": self._generate_processing_summary(results),
                "business_rationale": self._generate_business_rationale(applied_rules, context),
                "compliance_status": self._check_compliance_status(results),
                "recommendations": self._generate_rule_recommendations(results, context)
            }
            
            return explanation
            
        except Exception as e:
            logger.error(f"Errore generazione spiegazioni regole: {e}")
            return {"error": str(e)}
    
    def _generate_processing_summary(self, results: List[RuleResult]) -> str:
        """Genera riassunto processamento regole"""
        if not results:
            return "Nessuna regola applicata"
        
        applied = len([r for r in results if r.applied])
        
        if applied == 0:
            return f"Valutate {len(results)} regole, nessuna applicata (tutte le condizioni non soddisfatte)"
        
        modifications = len([r for r in results if r.applied and r.modified_value is not None])
        alerts = len([r for r in results if r.applied and r.action_taken == RuleAction.ALERT_ONLY])
        
        summary = f"Applicate {applied}/{len(results)} regole"
        if modifications > 0:
            summary += f", {modifications} modifiche al valore"
        if alerts > 0:
            summary += f", {alerts} alert generati"
        
        return summary
    
    def _generate_business_rationale(self, applied_rules: List[RuleResult], context: BusinessContext) -> str:
        """Genera rationale business per regole applicate"""
        if not applied_rules:
            return "Nessuna regola business applicata - valore forecast accettato come generato dal modello"
        
        rationale_parts = []
        
        # Analizza tipi di regole applicate
        constraint_rules = [r for r in applied_rules if self.rules[r.rule_id].rule_type == RuleType.CONSTRAINT]
        adjustment_rules = [r for r in applied_rules if self.rules[r.rule_id].rule_type == RuleType.ADJUSTMENT]
        validation_rules = [r for r in applied_rules if self.rules[r.rule_id].rule_type == RuleType.VALIDATION]
        
        if constraint_rules:
            rationale_parts.append(f"Applicati {len(constraint_rules)} vincoli operativi per rispettare limiti di capacità e budget")
        
        if adjustment_rules:
            rationale_parts.append(f"Applicate {len(adjustment_rules)} regole di aggiustamento per contesto business specifico")
        
        if validation_rules:
            rationale_parts.append(f"Applicate {len(validation_rules)} validazioni per garantire coerenza logica")
        
        # Contesto specifico
        context_info = []
        if context.is_weekend:
            context_info.append("periodo weekend")
        if context.is_holiday:
            context_info.append("periodo festivo")
        if context.season:
            context_info.append(f"stagione {context.season}")
        
        if context_info:
            rationale_parts.append(f"Considerato contesto: {', '.join(context_info)}")
        
        return ". ".join(rationale_parts) + "."
    
    def _check_compliance_status(self, results: List[RuleResult]) -> Dict[str, Any]:
        """Controlla status compliance regole"""
        return {
            "all_rules_passed": all(r.action_taken != RuleAction.REJECT for r in results),
            "violations": [r.rule_name for r in results if r.action_taken == RuleAction.REJECT],
            "pending_approvals": len(self.approval_queue),
            "compliance_score": len([r for r in results if r.applied and r.action_taken != RuleAction.REJECT]) / max(len(results), 1)
        }
    
    def _generate_rule_recommendations(self, results: List[RuleResult], context: BusinessContext) -> List[str]:
        """Genera raccomandazioni basate su risultati regole"""
        recommendations = []
        
        # Alert recommendations
        alerts = [r for r in results if r.action_taken == RuleAction.ALERT_ONLY]
        if alerts:
            recommendations.append(f"Monitorare {len(alerts)} alert generati per potenziali anomalie")
        
        # Modification recommendations
        modifications = [r for r in results if r.modified_value is not None]
        if len(modifications) > 3:
            recommendations.append("Numero elevato di modifiche - considerare revisione regole o modello")
        
        # Context-specific recommendations
        if context.max_capacity and any("capacità" in r.message.lower() for r in results):
            recommendations.append("Valutare espansione capacità se forecast continua a essere limitato")
        
        if not recommendations:
            recommendations.append("Nessuna raccomandazione specifica - monitoraggio standard")
        
        return recommendations
    
    def get_rules_summary(self) -> Dict[str, Any]:
        """Riassunto configurazione regole"""
        return {
            "total_rules": len(self.rules),
            "enabled_rules": len([r for r in self.rules.values() if r.enabled]),
            "rules_by_type": {
                rule_type.value: len([r for r in self.rules.values() if r.rule_type == rule_type])
                for rule_type in RuleType
            },
            "rules_by_priority": {
                f"priority_{p}": len([r for r in self.rules.values() if r.priority == p])
                for p in sorted(set(r.priority for r in self.rules.values()))
            },
            "most_applied_rules": sorted(
                [(r.name, r.application_count) for r in self.rules.values()],
                key=lambda x: x[1],
                reverse=True
            )[:5]
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """Statistiche engine regole"""
        recent_results = [r for r in self.rule_history if r.timestamp > datetime.now() - timedelta(days=7)]
        
        return {
            "evaluations_count": self.evaluations_count,
            "modifications_count": self.modifications_count,
            "alerts_count": self.alerts_count,
            "total_rules": len(self.rules),
            "enabled_rules": len([r for r in self.rules.values() if r.enabled]),
            "recent_applications": len(recent_results),
            "approval_queue_size": len(self.approval_queue),
            "modification_rate": (self.modifications_count / max(self.evaluations_count, 1)) * 100,
            "alert_rate": (self.alerts_count / max(self.evaluations_count, 1)) * 100
        }


# Utility functions
def create_inventory_context(
    product_id: str,
    forecast_date: datetime,
    capacity_limits: Tuple[float, float] = None,
    is_weekend: bool = None,
    category: str = "standard"
) -> BusinessContext:
    """Utility per creare contesto inventory management"""
    
    if is_weekend is None:
        is_weekend = forecast_date.weekday() >= 5
    
    context = BusinessContext(
        model_id=f"inventory_{product_id}",
        product_id=product_id,
        category=category,
        forecast_date=forecast_date,
        is_weekend=is_weekend
    )
    
    if capacity_limits:
        context.min_capacity, context.max_capacity = capacity_limits
    
    return context


def apply_business_rules_to_forecast(
    forecast_value: float,
    context: BusinessContext,
    rules_engine: BusinessRulesEngine = None
) -> Dict[str, Any]:
    """
    Utility per applicare regole business a forecast
    
    Args:
        forecast_value: Valore forecast originale
        context: Contesto business
        rules_engine: Engine regole (crea default se None)
        
    Returns:
        Dict con risultati applicazione regole
    """
    if rules_engine is None:
        rules_engine = BusinessRulesEngine()
    
    # Applica regole
    final_value, results = rules_engine.apply_rules(forecast_value, context)
    
    # Genera spiegazioni
    explanations = rules_engine.get_rule_explanations(forecast_value, context, results)
    
    return {
        "original_value": forecast_value,
        "final_value": final_value,
        "modification_applied": final_value != forecast_value,
        "modification_percentage": ((final_value - forecast_value) / forecast_value * 100) if forecast_value != 0 else 0,
        "rules_results": [r.dict() for r in results],
        "explanations": explanations,
        "context": context.dict()
    }


if __name__ == "__main__":
    # Test business rules engine
    print("Test Business Rules Engine")
    
    # Crea engine
    engine = BusinessRulesEngine()
    
    # Context di test
    context = BusinessContext(
        model_id="test_product",
        product_id="PROD-001",
        category="B2B",
        forecast_date=datetime(2024, 1, 15, 9, 0),  # Lunedì
        is_weekend=False,
        max_capacity=1000,
        historical_average=800
    )
    
    # Test valore normale
    print("\n=== Test 1: Valore Normale ===")
    result = apply_business_rules_to_forecast(850, context, engine)
    print(f"Original: {result['original_value']}")
    print(f"Final: {result['final_value']}")
    print(f"Modification: {result['modification_applied']}")
    
    # Test capacità superata
    print("\n=== Test 2: Capacità Superata ===")
    result = apply_business_rules_to_forecast(1200, context, engine)
    print(f"Original: {result['original_value']}")
    print(f"Final: {result['final_value']}")
    print(f"Rules applied: {len([r for r in result['rules_results'] if r['applied']])}")
    
    # Test weekend B2B
    print("\n=== Test 3: Weekend B2B ===")
    weekend_context = context.copy()
    weekend_context.is_weekend = True
    result = apply_business_rules_to_forecast(800, weekend_context, engine)
    print(f"Original: {result['original_value']}")
    print(f"Final: {result['final_value']}")
    print(f"Weekend reduction applied: {result['modification_applied']}")
    
    # Statistiche engine
    print(f"\n=== Statistiche Engine ===")
    stats = engine.get_stats()
    print(f"Evaluations: {stats['evaluations_count']}")
    print(f"Modifications: {stats['modifications_count']}")
    print(f"Enabled rules: {stats['enabled_rules']}")
    print(f"Modification rate: {stats['modification_rate']:.1f}%")