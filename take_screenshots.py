"""Take screenshots of the deployed GCM web app showing all 10 improvements."""
from playwright.sync_api import sync_playwright

SCREENSHOTS_DIR = '/home/user/weatherflow/screenshots'
CHROMIUM = '/root/.cache/ms-playwright/chromium_headless_shell-1194/chrome-linux/headless_shell'

def take_screenshots():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True, executable_path=CHROMIUM)
        page = browser.new_page(viewport={'width': 1400, 'height': 900})

        # Screenshot 1: Main page light mode (improvements 1, 5, 6, 9, 10)
        page.goto('http://localhost:5000')
        page.wait_for_timeout(1500)
        page.screenshot(path=f'{SCREENSHOTS_DIR}/01_main_page_light.png', full_page=True)
        print('1: Main page (light mode) - shows dark mode toggle, shortcut hint, health badge')

        # Screenshot 2: Dark mode (Improvement 1)
        page.click('#theme-toggle')
        page.wait_for_timeout(500)
        page.screenshot(path=f'{SCREENSHOTS_DIR}/02_dark_mode.png', full_page=True)
        print('2: Dark mode toggle (Improvement 1)')

        # Switch back to light
        page.click('#theme-toggle')
        page.wait_for_timeout(300)

        # Screenshot 3: Input validation (Improvement 5)
        page.fill('#co2_ppmv', '5000')
        page.fill('#duration_days', '200')
        page.click('#run-button')
        page.wait_for_timeout(500)
        page.screenshot(path=f'{SCREENSHOTS_DIR}/03_input_validation.png', full_page=False)
        print('3: Input validation with error messages (Improvement 5)')

        # Fix inputs and run simulation with fast settings
        page.fill('#co2_ppmv', '420')
        page.fill('#duration_days', '3')
        page.select_option('#nlon', '32')
        page.select_option('#nlat', '16')
        page.select_option('#nlev', '10')
        page.select_option('#dt', '1200')

        # Screenshot 4: Running with timer and progress animation (Improvements 4, 10)
        page.click('#run-button')
        page.wait_for_timeout(4000)
        page.screenshot(path=f'{SCREENSHOTS_DIR}/04_simulation_running.png', full_page=False)
        print('4: Simulation running - elapsed timer & progress animation (Improvements 4, 10)')

        # Wait for completion
        for i in range(120):
            status = page.text_content('#status-text')
            if 'complete' in status.lower():
                break
            page.wait_for_timeout(2000)

        page.wait_for_timeout(3000)

        # Screenshot 5: Results with export/compare buttons (Improvements 2, 7)
        page.screenshot(path=f'{SCREENSHOTS_DIR}/05_results_actions.png', full_page=False)
        print('5: Results panel with Export JSON & Compare buttons (Improvements 2, 7)')

        # Screenshot 6: Temperature Profile (Improvement 3)
        page.click('.tab-btn[data-plot="temp_profile"]')
        page.wait_for_timeout(2500)
        page.screenshot(path=f'{SCREENSHOTS_DIR}/06_temp_profile.png', full_page=False)
        print('6: Vertical temperature profile plot (Improvement 3)')

        # Screenshot 7: Cross-section (Improvement 8)
        page.click('.tab-btn[data-plot="cross_section"]')
        page.wait_for_timeout(2500)
        page.screenshot(path=f'{SCREENSHOTS_DIR}/07_cross_section.png', full_page=False)
        print('7: Pressure-level cross-section plot (Improvement 8)')

        # Screenshot 8: Diagnostics
        page.click('.tab-btn[data-plot="diagnostics"]')
        page.wait_for_timeout(2500)
        page.screenshot(path=f'{SCREENSHOTS_DIR}/08_diagnostics.png', full_page=False)
        print('8: Diagnostics 4-panel plot')

        # Screenshot 9: Health endpoint (Improvement 9)
        page2 = browser.new_page(viewport={'width': 900, 'height': 400})
        page2.goto('http://localhost:5000/api/health')
        page2.wait_for_timeout(500)
        page2.screenshot(path=f'{SCREENSHOTS_DIR}/09_health_endpoint.png')
        print('9: Health check API endpoint (Improvement 9)')
        page2.close()

        # Screenshot 10: Footer with uptime badge
        page.evaluate('window.scrollTo(0, document.body.scrollHeight)')
        page.wait_for_timeout(500)
        page.screenshot(path=f'{SCREENSHOTS_DIR}/10_footer_uptime.png', full_page=True)
        print('10: Full page with footer health badge (Improvement 9)')

        # Run 2nd simulation for comparison
        page.evaluate('window.scrollTo(0, 0)')
        page.wait_for_timeout(300)
        page.select_option('#profile', 'polar')
        page.fill('#co2_ppmv', '560')
        page.click('#run-button')

        for i in range(120):
            status = page.text_content('#status-text')
            if 'complete' in status.lower():
                break
            page.wait_for_timeout(2000)

        page.wait_for_timeout(3000)

        # Screenshot 11: Comparison modal (Improvement 7)
        btns = page.query_selector_all('button.action-btn')
        for btn in btns:
            if 'Compare' in (btn.text_content() or ''):
                btn.click()
                break

        page.wait_for_timeout(1000)
        compare_btn = page.query_selector('.compare-sim-btn')
        if compare_btn:
            compare_btn.click()
            page.wait_for_timeout(4000)
            page.screenshot(path=f'{SCREENSHOTS_DIR}/11_comparison_mode.png', full_page=False)
            print('11: Comparison mode with diff plot (Improvement 7)')

        # Close modal
        page.click('#compare-modal', position={'x': 10, 'y': 10})
        page.wait_for_timeout(300)

        # Screenshot 12: Export JSON (Improvement 2)
        sims = page.evaluate('() => Object.keys(window._simCache)')
        if sims:
            page3 = browser.new_page(viewport={'width': 900, 'height': 600})
            page3.goto(f'http://localhost:5000/api/export/{sims[-1]}')
            page3.wait_for_timeout(500)
            page3.screenshot(path=f'{SCREENSHOTS_DIR}/12_export_json.png')
            print('12: Exported JSON results (Improvement 2)')
            page3.close()

        # Screenshot 13: Keyboard shortcut hint visible (Improvement 6)
        page.evaluate('window.scrollTo(0, 0)')
        page.wait_for_timeout(300)
        # Zoom into the config panel area
        page.screenshot(path=f'{SCREENSHOTS_DIR}/13_keyboard_shortcut.png',
                       clip={'x': 0, 'y': 600, 'width': 450, 'height': 250})
        print('13: Keyboard shortcut hint (Improvement 6)')

        browser.close()
        print('\nAll screenshots saved!')

take_screenshots()
