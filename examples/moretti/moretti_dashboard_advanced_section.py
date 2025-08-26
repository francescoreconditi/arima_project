# Advanced Exog section to be added to Moretti dashboard

with tab7:
    st.subheader("🔬 Advanced Exogenous Analysis")
    
    if not FORECASTING_AVAILABLE:
        st.warning("⚠️ Advanced forecasting modules not available. Install full ARIMA Forecaster package.")
    else:
        # Info box
        st.info("""
        **Advanced Exogenous Analysis** consente di utilizzare variabili esterne (exogenous)
        per migliorare la precisione delle previsioni SARIMAX. Analizza relazioni, preprocessa
        i dati e seleziona automaticamente le features più rilevanti.
        """)
        
        # Tabs per diverse analisi
        subtab1, subtab2, subtab3, subtab4 = st.tabs([
            "📊 Relationships",
            "🔧 Preprocessing", 
            "🔍 Diagnostics",
            "🎯 Feature Selection"
        ])
        
        with subtab1:
            st.markdown("### 📊 Feature Relationships Analysis")
            
            # Simula alcune variabili exog per demo
            np.random.seed(42)
            n_days = len(vendite)
            exog_demo = pd.DataFrame({
                'temperatura': np.random.normal(20, 5, n_days),
                'promocioni': np.random.binomial(1, 0.3, n_days),
                'festivi': np.random.binomial(1, 0.1, n_days),
                'marketing_spend': np.random.exponential(1000, n_days)
            }, index=vendite.index)
            
            st.markdown("**Demo Exogenous Variables:**")
            st.dataframe(exog_demo.head(10), use_container_width=True)
            
            if prodotto_sel != 'Tutti':
                try:
                    # Analisi relazioni
                    target_series = vendite[prodotto_sel]
                    
                    st.markdown("**📈 Correlation Analysis:**")
                    corr_data = []
                    for feature in exog_demo.columns:
                        corr = target_series.corr(exog_demo[feature])
                        corr_data.append({
                            'Feature': feature,
                            'Correlation': f"{corr:.3f}",
                            'Strength': 'Strong' if abs(corr) > 0.7 else 'Moderate' if abs(corr) > 0.3 else 'Weak'
                        })
                    
                    corr_df = pd.DataFrame(corr_data)
                    st.dataframe(corr_df, use_container_width=True)
                    
                    # Grafico correlazioni
                    fig_corr = go.Figure()
                    fig_corr.add_trace(go.Bar(
                        x=exog_demo.columns,
                        y=[target_series.corr(exog_demo[col]) for col in exog_demo.columns],
                        marker_color=['green' if abs(target_series.corr(exog_demo[col])) > 0.5 else 'orange' for col in exog_demo.columns]
                    ))
                    fig_corr.update_layout(
                        title=f'Feature Correlations with {prodotto_sel}',
                        xaxis_title='Exogenous Variables',
                        yaxis_title='Correlation Coefficient'
                    )
                    st.plotly_chart(fig_corr, use_container_width=True)
                    
                except Exception as e:
                    st.error(f"Error in relationship analysis: {e}")
            else:
                st.info("Seleziona un prodotto specifico per l'analisi delle relazioni.")
        
        with subtab2:
            st.markdown("### 🔧 Advanced Preprocessing")
            
            # Configurazione preprocessing
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Outlier Detection:**")
                outlier_method = st.selectbox(
                    "Method:",
                    ['iqr', 'zscore', 'modified_zscore', 'isolation_forest'],
                    help="Metodo per il rilevamento degli outlier"
                )
                
                missing_strategy = st.selectbox(
                    "Missing Values Strategy:",
                    ['mean', 'median', 'interpolate', 'forward_fill', 'drop'],
                    help="Strategia per gestire valori mancanti"
                )
            
            with col2:
                st.markdown("**Feature Engineering:**")
                multicollinearity_threshold = st.slider(
                    "Multicollinearity Threshold:",
                    0.7, 0.99, 0.9,
                    help="Soglia per rilevare multicollinearità tra features"
                )
                
                apply_log_transform = st.checkbox(
                    "Apply Log Transform",
                    help="Applica trasformazione logaritmica alle variabili positive"
                )
            
            if st.button("🔧 Run Preprocessing Analysis", type="primary"):
                with st.spinner("Running preprocessing analysis..."):
                    try:
                        # Configura preprocessor
                        preprocessor = ExogenousPreprocessor(
                            outlier_method=outlier_method,
                            missing_strategy=missing_strategy,
                            multicollinearity_threshold=multicollinearity_threshold
                        )
                        
                        # Applica preprocessing ai dati demo
                        exog_processed, summary = preprocessor.fit_transform_with_summary(exog_demo)
                        
                        # Mostra risultati
                        st.success("✅ Preprocessing completato!")
                        
                        # Summary preprocessing
                        st.markdown("**📋 Preprocessing Summary:**")
                        for key, value in summary.items():
                            if isinstance(value, (int, float)):
                                st.metric(key.replace('_', ' ').title(), value)
                            else:
                                st.write(f"**{key.replace('_', ' ').title()}:** {value}")
                        
                        # Confronto before/after
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown("**Before Preprocessing:**")
                            st.dataframe(exog_demo.describe(), use_container_width=True)
                        
                        with col2:
                            st.markdown("**After Preprocessing:**")
                            st.dataframe(exog_processed.describe(), use_container_width=True)
                    
                    except Exception as e:
                        st.error(f"Preprocessing error: {e}")
        
        with subtab3:
            st.markdown("### 🔍 Advanced Diagnostics")
            
            if prodotto_sel != 'Tutti':
                try:
                    # Inizializza diagnostics
                    diagnostics = ExogDiagnostics()
                    
                    target_series = vendite[prodotto_sel]
                    
                    # Granger Causality Test
                    st.markdown("**🔬 Granger Causality Analysis:**")
                    
                    causality_results = []
                    for feature in exog_demo.columns:
                        try:
                            p_value = diagnostics.granger_causality_test(
                                target_series, 
                                exog_demo[feature],
                                maxlag=7
                            )
                            causality_results.append({
                                'Feature': feature,
                                'P-Value': f"{p_value:.4f}",
                                'Significant': 'Yes' if p_value < 0.05 else 'No',
                                'Causality': 'Strong' if p_value < 0.01 else 'Moderate' if p_value < 0.05 else 'Weak'
                            })
                        except Exception:
                            causality_results.append({
                                'Feature': feature,
                                'P-Value': 'N/A',
                                'Significant': 'N/A',
                                'Causality': 'N/A'
                            })
                    
                    causality_df = pd.DataFrame(causality_results)
                    st.dataframe(causality_df, use_container_width=True)
                    
                    # Feature Importance (Correlation-based)
                    st.markdown("**🎯 Feature Importance Ranking:**")
                    
                    importance_data = []
                    for feature in exog_demo.columns:
                        corr = abs(target_series.corr(exog_demo[feature]))
                        importance_data.append({
                            'Feature': feature,
                            'Importance Score': f"{corr:.4f}",
                            'Rank': 0  # Will be set below
                        })
                    
                    # Sort by importance and assign ranks
                    importance_df = pd.DataFrame(importance_data)
                    importance_df = importance_df.sort_values('Importance Score', ascending=False)
                    importance_df['Rank'] = range(1, len(importance_df) + 1)
                    
                    st.dataframe(importance_df, use_container_width=True)
                    
                except Exception as e:
                    st.error(f"Diagnostics error: {e}")
            else:
                st.info("Seleziona un prodotto specifico per i diagnostics avanzati.")
        
        with subtab4:
            st.markdown("### 🎯 Automatic Feature Selection")
            
            if prodotto_sel != 'Tutti':
                # Configurazione feature selection
                col1, col2 = st.columns(2)
                
                with col1:
                    selection_method = st.selectbox(
                        "Feature Selection Method:",
                        ['stepwise', 'lasso', 'elastic_net', 'f_test'],
                        help="Metodo di selezione automatica delle features"
                    )
                    
                    seasonal_order = st.selectbox(
                        "Seasonal Order (s):",
                        [7, 12, 24, 30],
                        index=0,
                        help="Periodo stagionale per SARIMAX"
                    )
                
                with col2:
                    max_features = st.slider(
                        "Max Features to Select:",
                        1, len(exog_demo.columns), min(3, len(exog_demo.columns)),
                        help="Numero massimo di features da selezionare"
                    )
                    
                    test_size = st.slider(
                        "Test Size (%):",
                        10, 50, 20,
                        help="Percentuale dati per test set"
                    ) / 100
                
                if st.button("🚀 Run Advanced SARIMAX Analysis", type="primary"):
                    with st.spinner("Running SARIMAX Auto-Selection..."):
                        try:
                            # Prepara i dati
                            target_series = vendite[prodotto_sel].asfreq('D').fillna(0)
                            exog_data = exog_demo.copy()
                            
                            # Split train/test
                            split_idx = int(len(target_series) * (1 - test_size))
                            y_train = target_series[:split_idx]
                            y_test = target_series[split_idx:]
                            X_train = exog_data[:split_idx]
                            X_test = exog_data[split_idx:]
                            
                            # Configura SARIMAXAutoSelector
                            auto_selector = SARIMAXAutoSelector(
                                feature_selection_method=selection_method,
                                max_features=max_features,
                                seasonal_order=(1, 1, 1, seasonal_order)
                            )
                            
                            # Fit con feature selection automatica
                            auto_selector.fit_with_exog(
                                y_train, 
                                X_train,
                                validate_input=False
                            )
                            
                            # Risultati feature selection
                            st.success("✅ SARIMAX Auto-Selection completata!")
                            
                            # Mostra features selezionate
                            selected_features = auto_selector.selected_features_
                            st.markdown("**🎯 Selected Features:**")
                            
                            if selected_features:
                                for i, feature in enumerate(selected_features, 1):
                                    st.write(f"{i}. **{feature}**")
                            else:
                                st.warning("Nessuna feature selezionata automaticamente.")
                            
                            # Performance metrics
                            st.markdown("**📊 Model Performance:**")
                            
                            try:
                                # Generate forecasts
                                forecast = auto_selector.forecast_with_exog(
                                    X_test, steps=len(y_test)
                                )
                                
                                # Calculate metrics
                                mae = np.mean(np.abs(y_test - forecast))
                                rmse = np.sqrt(np.mean((y_test - forecast) ** 2))
                                mape = np.mean(np.abs((y_test - forecast) / np.maximum(y_test, 1))) * 100
                                
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.metric("MAE", f"{mae:.3f}")
                                with col2:
                                    st.metric("RMSE", f"{rmse:.3f}")
                                with col3:
                                    st.metric("MAPE", f"{mape:.2f}%")
                                
                                # Grafico forecast vs actual
                                fig_forecast = go.Figure()
                                
                                # Actual values
                                fig_forecast.add_trace(go.Scatter(
                                    x=y_test.index,
                                    y=y_test.values,
                                    mode='lines+markers',
                                    name='Actual',
                                    line=dict(color='blue')
                                ))
                                
                                # Forecasted values
                                fig_forecast.add_trace(go.Scatter(
                                    x=y_test.index,
                                    y=forecast,
                                    mode='lines+markers',
                                    name='SARIMAX Forecast',
                                    line=dict(color='red', dash='dash')
                                ))
                                
                                fig_forecast.update_layout(
                                    title=f'SARIMAX Advanced Forecast - {prodotto_sel}',
                                    xaxis_title='Date',
                                    yaxis_title='Sales Units',
                                    hovermode='x unified'
                                )
                                
                                st.plotly_chart(fig_forecast, use_container_width=True)
                                
                            except Exception as forecast_error:
                                st.warning(f"Forecast generation error: {forecast_error}")
                            
                        except Exception as e:
                            st.error(f"SARIMAX Auto-Selection error: {e}")
                            st.error("Ensure you have sufficient historical data and valid exogenous variables.")
            else:
                st.info("Seleziona un prodotto specifico per la feature selection avanzata.")