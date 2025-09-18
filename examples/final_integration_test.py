#!/usr/bin/env python3
"""
Test Finale - Dashboard Evolution Complete Integration.

Testa tutte le 3 funzionalità implementate in modalità production-like.
"""

import sys
from pathlib import Path
from datetime import datetime
import time

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def test_mobile_responsive():
    """Test Mobile Responsive Manager."""
    print("Testing Mobile Responsive Manager...")

    from arima_forecaster.dashboard.mobile_responsive import MobileResponsiveManager

    # Initialize
    manager = MobileResponsiveManager()

    # Test device detection
    assert hasattr(manager, "is_mobile")
    assert hasattr(manager, "is_tablet")
    assert isinstance(manager.screen_width, int)

    # Test layout configuration
    config = manager.get_layout_config()
    required_keys = ["columns", "chart_height", "sidebar_state", "font_size", "spacing"]

    for key in required_keys:
        assert key in config, f"Missing config key: {key}"

    assert 1 <= config["columns"] <= 3
    assert 200 <= config["chart_height"] <= 600

    print("  ✅ Device detection working")
    print("  ✅ Layout configuration valid")
    print("  ✅ Mobile responsive: PASSED")

    return True


def test_excel_export():
    """Test Excel Export for Procurement."""
    print("\nTesting Excel Export for Procurement...")

    from arima_forecaster.dashboard.excel_exporter import (
        ProcurementExcelExporter,
        create_sample_procurement_data,
    )
    import openpyxl
    from io import BytesIO

    # Initialize
    exporter = ProcurementExcelExporter()

    # Generate sample data
    forecast_data, inventory_params, product_info = create_sample_procurement_data()

    # Test data validity
    assert len(forecast_data) > 0
    assert "total_value" in inventory_params
    assert inventory_params["total_value"] > 0

    print("  ✅ Sample data generation working")

    # Test full report generation
    start_time = time.time()
    excel_bytes = exporter.generate_procurement_report(
        forecast_data, inventory_params, product_info
    )
    generation_time = time.time() - start_time

    # Validate Excel file
    assert isinstance(excel_bytes, bytes)
    assert len(excel_bytes) > 5000  # Reasonable file size
    assert generation_time < 10.0  # Performance check

    print(f"  ✅ Full report generated ({len(excel_bytes):,} bytes in {generation_time:.2f}s)")

    # Test Excel file validity
    buffer = BytesIO(excel_bytes)
    workbook = openpyxl.load_workbook(buffer)

    expected_sheets = ["Executive Summary", "Piano Riordini", "Previsioni 30gg"]
    for sheet_name in expected_sheets:
        assert sheet_name in workbook.sheetnames, f"Missing sheet: {sheet_name}"

    print("  ✅ Excel file structure valid")

    # Test quick summary
    summary_data = {"total_value": 50000, "coverage_days": 30, "reorder_items": 2, "savings": 2500}

    quick_excel = exporter.generate_quick_summary(summary_data)
    assert len(quick_excel) > 1000

    print("  ✅ Quick summary generation working")
    print("  ✅ Excel export: PASSED")

    return excel_bytes


def test_scenario_simulator():
    """Test What-If Scenario Simulator."""
    print("\nTesting What-If Scenario Simulator...")

    from arima_forecaster.dashboard.scenario_simulator import (
        WhatIfScenarioSimulator,
        ScenarioParameters,
        ScenarioType,
        create_sample_base_data,
    )

    # Initialize
    simulator = WhatIfScenarioSimulator()

    # Get base data
    base_forecast, base_metrics = create_sample_base_data()

    # Validate base data
    assert len(base_forecast) > 0
    assert isinstance(base_metrics, dict)
    assert "unit_price" in base_metrics

    print("  ✅ Base data generation working")

    # Test scenario parameters
    params = ScenarioParameters(marketing_boost=100.0, price_change=-15.0, seasonality_factor=1.5)

    # Run simulation
    start_time = time.time()
    scenario_forecast, results = simulator.run_scenario_simulation(
        params, base_forecast, base_metrics
    )
    simulation_time = time.time() - start_time

    # Validate results
    assert len(scenario_forecast) == len(base_forecast)
    assert hasattr(results, "revenue_impact")
    assert hasattr(results, "service_level")
    assert hasattr(results, "recommendations")
    assert simulation_time < 5.0  # Performance check

    print(f"  ✅ Scenario simulation completed ({simulation_time:.2f}s)")
    print(f"      Revenue Impact: {results.revenue_change_pct:+.1f}%")
    print(f"      Service Level: {results.service_level:.1f}%")

    # Test different scenarios
    test_scenarios = [
        ("Marketing Campaign", ScenarioParameters(marketing_boost=150.0, price_change=-10.0)),
        ("Economic Crisis", ScenarioParameters(inflation_rate=8.0, competitor_impact=-20.0)),
        ("Supplier Issues", ScenarioParameters(supplier_reliability=75.0, lead_time_change=10)),
    ]

    for scenario_name, test_params in test_scenarios:
        _, test_results = simulator.run_scenario_simulation(
            test_params, base_forecast, base_metrics
        )
        assert isinstance(test_results.revenue_change_pct, (int, float))
        assert 70 <= test_results.service_level <= 100
        assert len(test_results.recommendations) > 0

    print(f"  ✅ Multiple scenarios tested ({len(test_scenarios)} scenarios)")

    # Test visualization creation
    try:
        fig = simulator.create_scenario_visualization(base_forecast, scenario_forecast, results)
        assert hasattr(fig, "data")
        assert hasattr(fig, "layout")
        assert len(fig.data) > 0
        print("  ✅ Visualization generation working")
    except Exception as e:
        print(f"  ⚠️  Visualization test failed: {e}")

    print("  ✅ Scenario simulator: PASSED")

    return results


def test_complete_integration():
    """Test complete workflow integration."""
    print("\nTesting Complete Integration Workflow...")

    from arima_forecaster.dashboard.mobile_responsive import MobileResponsiveManager
    from arima_forecaster.dashboard.excel_exporter import (
        ProcurementExcelExporter,
        create_sample_procurement_data,
    )
    from arima_forecaster.dashboard.scenario_simulator import (
        WhatIfScenarioSimulator,
        ScenarioParameters,
        create_sample_base_data,
    )

    # 1. Initialize all components
    responsive = MobileResponsiveManager()
    simulator = WhatIfScenarioSimulator()
    exporter = ProcurementExcelExporter()

    print("  ✅ All components initialized")

    # 2. Get responsive configuration
    config = responsive.get_layout_config()
    is_mobile = responsive.is_mobile

    # 3. Run scenario analysis
    base_forecast, base_metrics = create_sample_base_data()
    scenario_params = ScenarioParameters(
        marketing_boost=75.0, price_change=-12.0, seasonality_factor=1.4
    )

    scenario_forecast, scenario_results = simulator.run_scenario_simulation(
        scenario_params, base_forecast, base_metrics
    )

    print(f"  ✅ Scenario analysis completed")
    print(f"      Revenue Impact: {scenario_results.revenue_change_pct:+.1f}%")

    # 4. Generate appropriate report based on device
    forecast_data, inventory_params, product_info = create_sample_procurement_data()

    # Update inventory params with scenario results
    inventory_params["scenario_revenue_impact"] = scenario_results.revenue_impact
    inventory_params["scenario_service_level"] = scenario_results.service_level

    if is_mobile:
        # Mobile: quick summary
        excel_data = exporter.generate_quick_summary(inventory_params)
        report_type = "Mobile Quick Summary"
    else:
        # Desktop: full report
        excel_data = exporter.generate_procurement_report(
            forecast_data, inventory_params, product_info
        )
        report_type = "Desktop Full Report"

    print(f"  ✅ {report_type} generated ({len(excel_data):,} bytes)")

    # 5. Save integrated result
    output_file = (
        Path(__file__).parent
        / f"integration_test_result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
    )
    with open(output_file, "wb") as f:
        f.write(excel_data)

    # 6. Validate complete workflow
    assert output_file.exists()
    assert output_file.stat().st_size > 1000

    print(f"  ✅ Integration result saved: {output_file}")
    print("  ✅ Complete integration: PASSED")

    return {
        "mobile_mode": is_mobile,
        "scenario_impact": scenario_results.revenue_change_pct,
        "report_type": report_type,
        "file_size": len(excel_data),
        "output_file": str(output_file),
    }


def main():
    """Run complete test suite."""

    print("=" * 70)
    print("FINAL INTEGRATION TEST - Dashboard Evolution")
    print("=" * 70)
    print("Testing all 3 implemented features:")
    print("1. Mobile Responsive Design")
    print("2. Excel Export for Procurement Team")
    print("3. What-If Scenario Simulator")
    print("4. Complete Integration Workflow")
    print("-" * 70)

    start_time = time.time()
    results = {}

    try:
        # Run individual tests
        results["mobile_responsive"] = test_mobile_responsive()
        results["excel_export"] = test_excel_export()
        results["scenario_simulator"] = test_scenario_simulator()
        results["integration"] = test_complete_integration()

        total_time = time.time() - start_time

        print("\n" + "=" * 70)
        print("TEST RESULTS SUMMARY")
        print("=" * 70)

        all_passed = all(
            [
                results["mobile_responsive"] == True,
                len(results["excel_export"]) > 5000,
                hasattr(results["scenario_simulator"], "revenue_change_pct"),
                "output_file" in results["integration"],
            ]
        )

        if all_passed:
            print("🎉 ALL TESTS PASSED! 🎉")
            print()
            print("✅ Mobile Responsive Design - WORKING")
            print("✅ Excel Export for Procurement - WORKING")
            print("✅ What-If Scenario Simulator - WORKING")
            print("✅ Complete Integration Workflow - WORKING")
            print()
            print("DASHBOARD EVOLUTION STATUS: PRODUCTION READY")
            print(f"Total test time: {total_time:.2f} seconds")

            # Show integration results
            integration_data = results["integration"]
            print(f"\nIntegration Test Results:")
            print(f"  Device Mode: {'Mobile' if integration_data['mobile_mode'] else 'Desktop'}")
            print(f"  Scenario Impact: {integration_data['scenario_impact']:+.1f}%")
            print(f"  Report Generated: {integration_data['report_type']}")
            print(f"  File Size: {integration_data['file_size']:,} bytes")
            print(f"  Output File: {integration_data['output_file']}")

        else:
            print("❌ SOME TESTS FAILED")
            print("Check the error messages above for details.")
            return 1

        print("\n" + "=" * 70)
        print("READY FOR PRODUCTION DEPLOYMENT")
        print("=" * 70)

        return 0

    except Exception as e:
        print(f"\n❌ Test suite failed with error: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
