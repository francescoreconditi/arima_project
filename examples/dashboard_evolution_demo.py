#!/usr/bin/env python3
"""
Demo Dashboard Evolution - Mobile, Excel Export, What-If Simulator.

Dimostra tutte e 3 le nuove funzionalità enterprise implementate.
"""

import sys
from pathlib import Path
from datetime import datetime
import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from arima_forecaster.dashboard.mobile_responsive import MobileResponsiveManager
from arima_forecaster.dashboard.excel_exporter import (
    ProcurementExcelExporter,
    create_sample_procurement_data,
)
from arima_forecaster.dashboard.scenario_simulator import (
    WhatIfScenarioSimulator,
    ScenarioParameters,
    ScenarioType,
    create_sample_base_data,
)


def demo_mobile_responsive():
    """Demo Mobile Responsive Design."""

    print("\n" + "=" * 60)
    print("[MOBILE] DEMO: Mobile Responsive Design")
    print("=" * 60)

    # Initialize responsive manager
    responsive = MobileResponsiveManager()

    print(f"Screen Width: {responsive.screen_width}px")
    print(f"Is Mobile: {responsive.is_mobile}")
    print(f"Is Tablet: {responsive.is_tablet}")
    print()

    # Get layout configuration
    config = responsive.get_layout_config()
    print("Layout Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")

    print("\nResponsive Features:")
    print("  ✅ Auto-detect mobile/tablet devices")
    print("  ✅ Adaptive column layouts (1-3 columns)")
    print("  ✅ Dynamic chart heights (300-500px)")
    print("  ✅ Mobile-optimized CSS styling")
    print("  ✅ Responsive navigation components")

    return responsive


def demo_excel_export():
    """Demo Excel Export per Procurement."""

    print("\n" + "=" * 60)
    print("📊 DEMO: Excel Export per Procurement Team")
    print("=" * 60)

    # Initialize exporter
    exporter = ProcurementExcelExporter()

    # Generate sample data
    forecast_data, inventory_params, product_info = create_sample_procurement_data()

    print("Sample Data Generated:")
    print(f"  Forecast: {len(forecast_data)} days")
    print(f"  Inventory Value: €{inventory_params['total_value']:,}")
    print(f"  Product SKUs: {product_info['total_skus']}")
    print()

    # Generate full procurement report
    print("Generating Full Procurement Report...")
    excel_data = exporter.generate_procurement_report(
        forecast_data=forecast_data,
        inventory_params=inventory_params,
        product_info=product_info,
        supplier_data={},
    )

    # Save report
    output_file = (
        Path(__file__).parent / f"procurement_report_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx"
    )
    with open(output_file, "wb") as f:
        f.write(excel_data)

    print(f"✅ Full Report Generated: {len(excel_data):,} bytes")
    print(f"📁 Saved to: {output_file}")

    # Generate quick summary
    print("\nGenerating Quick Summary...")
    summary_data = {
        "total_value": inventory_params["total_value"],
        "coverage_days": inventory_params["coverage_days"],
        "reorder_items": 3,
        "savings": inventory_params["savings_potential"],
    }

    quick_excel = exporter.generate_quick_summary(summary_data)
    quick_file = (
        Path(__file__).parent / f"quick_summary_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx"
    )

    with open(quick_file, "wb") as f:
        f.write(quick_excel)

    print(f"✅ Quick Summary: {len(quick_excel):,} bytes")
    print(f"📁 Saved to: {quick_file}")

    print("\nExcel Report Features:")
    print("  📋 Executive Summary with KPIs")
    print("  🛒 Detailed Reorder Plan")
    print("  📈 30-day Forecast with CI")
    print("  📊 Performance Analysis")
    print("  ⚠️ Risk Assessment Matrix")
    print("  ✅ Action Items with Timeline")
    print("  🎨 Professional Formatting")

    return excel_data


def demo_scenario_simulator():
    """Demo What-If Scenario Simulator."""

    print("\n" + "=" * 60)
    print("🎯 DEMO: What-If Scenario Simulator")
    print("=" * 60)

    # Initialize simulator
    simulator = WhatIfScenarioSimulator()

    # Get base data
    base_forecast, base_metrics = create_sample_base_data()

    print(f"Base Forecast: {len(base_forecast)} days")
    print("Base Metrics:")
    for key, value in base_metrics.items():
        print(f"  {key}: {value}")
    print()

    # Test different scenarios
    scenarios_to_test = [
        {
            "name": "🚀 Marketing Campaign",
            "params": ScenarioParameters(
                marketing_boost=150.0, price_change=-10.0, seasonality_factor=1.2
            ),
        },
        {
            "name": "⚠️ Economic Crisis",
            "params": ScenarioParameters(
                marketing_boost=-20.0,
                inflation_rate=8.0,
                competitor_impact=-15.0,
                seasonality_factor=0.8,
            ),
        },
        {
            "name": "🏭 Supplier Issues",
            "params": ScenarioParameters(
                supplier_reliability=75.0, lead_time_change=14, capacity_limit=70.0
            ),
        },
        {
            "name": "🛍️ Black Friday",
            "params": ScenarioParameters(
                marketing_boost=300.0,
                price_change=-40.0,
                seasonality_factor=2.0,
                capacity_limit=80.0,
            ),
        },
    ]

    results_summary = []

    print("Running Scenario Simulations:")
    print("-" * 40)

    for scenario in scenarios_to_test:
        name = scenario["name"]
        params = scenario["params"]

        # Run simulation
        scenario_forecast, results = simulator.run_scenario_simulation(
            params, base_forecast, base_metrics
        )

        # Display results
        print(f"\n{name}:")
        print(f"  Revenue Impact: {results.revenue_change_pct:+.1f}%")
        print(f"  Service Level: {results.service_level:.1f}%")
        print(f"  ROI 3M: {results.roi_3months:+.1f}%")
        print(f"  Break Even: {results.break_even_days} giorni")

        if results.recommendations:
            print(f"  Top Recommendation: {results.recommendations[0]}")

        results_summary.append(
            {
                "scenario": name,
                "revenue_impact": results.revenue_change_pct,
                "service_level": results.service_level,
                "roi": results.roi_3months,
            }
        )

    print("\n" + "=" * 40)
    print("SCENARIO COMPARISON SUMMARY")
    print("=" * 40)

    # Create comparison table
    df = pd.DataFrame(results_summary)
    print(df.to_string(index=False, float_format="%.1f"))

    print("\nScenario Simulator Features:")
    print("  🎛️ Interactive Parameter Controls")
    print("  📊 Real-time Impact Calculation")
    print("  📈 Multi-chart Visualizations")
    print("  💡 Automated Recommendations")
    print("  🔄 Predefined Scenario Templates")
    print("  ⚖️ Risk vs Reward Analysis")

    return results_summary


def demo_integration():
    """Demo integrazione completa workflow."""

    print("\n" + "=" * 60)
    print("🔄 DEMO: Complete Integration Workflow")
    print("=" * 60)

    print("Simulating Complete Business Workflow...")
    print()

    # 1. Mobile-responsive setup
    print("1️⃣ Setting up mobile-responsive environment...")
    responsive = MobileResponsiveManager()
    config = responsive.get_layout_config()
    print(f"   📱 Device: {'Mobile' if responsive.is_mobile else 'Desktop'}")
    print(f"   📐 Layout: {config['columns']} columns")

    # 2. Scenario analysis
    print("\n2️⃣ Running scenario analysis...")
    simulator = WhatIfScenarioSimulator()
    base_forecast, base_metrics = create_sample_base_data()

    # Best case scenario
    best_case = ScenarioParameters(marketing_boost=75.0, price_change=-5.0, seasonality_factor=1.3)

    scenario_forecast, results = simulator.run_scenario_simulation(
        best_case, base_forecast, base_metrics
    )

    print(f"   📈 Revenue Impact: {results.revenue_change_pct:+.1f}%")
    print(f"   📊 Service Level: {results.service_level:.1f}%")

    # 3. Excel report generation
    print("\n3️⃣ Generating procurement reports...")
    exporter = ProcurementExcelExporter()
    forecast_data, inventory_params, product_info = create_sample_procurement_data()

    # Update params with scenario results
    inventory_params.update(
        {
            "scenario_revenue_impact": results.revenue_impact,
            "scenario_service_level": results.service_level,
            "scenario_recommendations": results.recommendations[:3],
        }
    )

    if responsive.is_mobile:
        # Mobile: quick summary
        excel_data = exporter.generate_quick_summary(inventory_params)
        report_type = "Quick Summary"
    else:
        # Desktop: full report
        excel_data = exporter.generate_procurement_report(
            forecast_data, inventory_params, product_info
        )
        report_type = "Full Report"

    print(f"   📋 Generated: {report_type} ({len(excel_data):,} bytes)")

    # 4. Export results
    final_file = (
        Path(__file__).parent / f"integrated_workflow_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx"
    )
    with open(final_file, "wb") as f:
        f.write(excel_data)

    print(f"   💾 Saved: {final_file}")

    print("\n✅ Complete Workflow Integration SUCCESS!")
    print("   🔄 Responsive Design → Scenario Analysis → Excel Export")
    print("   📱 Mobile/Desktop Optimized")
    print("   🎯 Business Decision Support")
    print("   📊 Professional Reporting")

    return {
        "responsive_config": config,
        "scenario_results": results,
        "excel_size": len(excel_data),
        "output_file": final_file,
    }


def main():
    """Run complete dashboard evolution demo."""

    print("[ROCKET] DASHBOARD EVOLUTION DEMO")
    print("Enterprise Features for ARIMA Forecaster Pro")
    print("=" * 60)
    print("Date:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    try:
        # Run individual demos
        demo_mobile_responsive()
        demo_excel_export()
        demo_scenario_simulator()

        # Run integration demo
        integration_results = demo_integration()

        print("\n" + "=" * 60)
        print("🎉 DEMO COMPLETED SUCCESSFULLY!")
        print("=" * 60)

        print("\n📋 Summary of Features Demonstrated:")
        print("  📱 Mobile Responsive Design - ✅")
        print("  📊 Excel Export for Procurement - ✅")
        print("  🎯 What-If Scenario Simulator - ✅")
        print("  🔄 Complete Integration Workflow - ✅")

        print(f"\n📁 Generated Files:")
        demo_dir = Path(__file__).parent
        excel_files = list(demo_dir.glob("*.xlsx"))

        for file in excel_files:
            if file.stat().st_mtime > (datetime.now().timestamp() - 3600):  # Created in last hour
                size = file.stat().st_size
                print(f"  📄 {file.name} ({size:,} bytes)")

        print(f"\n🎯 Business Value Demonstrated:")
        print("  💼 Enterprise-grade dashboard capabilities")
        print("  📱 Multi-device accessibility (mobile/tablet/desktop)")
        print("  📊 Professional reporting for procurement teams")
        print("  🎯 Interactive scenario planning for decision making")
        print("  🔄 Integrated workflow for complete business process")

        print(f"\n🚀 Next Steps for Production:")
        print("  1. Deploy enhanced dashboard to production server")
        print("  2. Configure real ERP data integration")
        print("  3. Setup automated report scheduling")
        print("  4. Train procurement team on new features")
        print("  5. Monitor usage analytics and user feedback")

        return integration_results

    except Exception as e:
        print(f"\n❌ Demo failed: {str(e)}")
        raise


if __name__ == "__main__":
    results = main()
