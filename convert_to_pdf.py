#!/usr/bin/env python3
"""
Convert HTML guide to PDF
"""

import os
import subprocess

def convert_html_to_pdf():
    """Convert HTML guide to PDF using wkhtmltopdf"""
    html_file = "/home/user/weatherflow/WeatherFlow_User_Guide.html"
    pdf_file = "/home/user/weatherflow/WeatherFlow_User_Guide.pdf"

    print("Converting HTML to PDF...")
    print(f"Input: {html_file}")
    print(f"Output: {pdf_file}")

    # Try different methods for PDF conversion
    methods = []

    # Method 1: Try wkhtmltopdf
    try:
        result = subprocess.run(
            ["which", "wkhtmltopdf"],
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            methods.append(("wkhtmltopdf", [
                "wkhtmltopdf",
                "--enable-local-file-access",
                "--page-size", "A4",
                "--margin-top", "20mm",
                "--margin-bottom", "20mm",
                "--margin-left", "20mm",
                "--margin-right", "20mm",
                "--print-media-type",
                html_file,
                pdf_file
            ]))
    except:
        pass

    # Method 2: Try weasyprint
    try:
        import weasyprint
        methods.append(("weasyprint", None))
    except ImportError:
        pass

    # Method 3: Try chromium/chrome headless
    for browser in ["chromium-browser", "chromium", "google-chrome", "chrome"]:
        try:
            result = subprocess.run(
                ["which", browser],
                capture_output=True,
                text=True
            )
            if result.returncode == 0:
                methods.append((browser, [
                    browser,
                    "--headless",
                    "--disable-gpu",
                    "--no-sandbox",
                    "--print-to-pdf=" + pdf_file,
                    html_file
                ]))
                break
        except:
            pass

    # Try each method
    success = False

    for method_name, command in methods:
        try:
            if method_name == "weasyprint":
                print(f"Trying {method_name}...")
                from weasyprint import HTML
                HTML(filename=html_file).write_pdf(pdf_file)
                success = True
                print(f"✅ Successfully converted using {method_name}")
                break
            else:
                print(f"Trying {method_name}...")
                result = subprocess.run(
                    command,
                    capture_output=True,
                    text=True,
                    timeout=300
                )
                if result.returncode == 0 and os.path.exists(pdf_file):
                    success = True
                    print(f"✅ Successfully converted using {method_name}")
                    break
                else:
                    print(f"❌ {method_name} failed: {result.stderr}")
        except Exception as e:
            print(f"❌ {method_name} error: {e}")

    if not success:
        print("\n⚠️ Could not convert to PDF using available methods.")
        print("Available alternatives:")
        print("1. Install wkhtmltopdf: sudo apt-get install wkhtmltopdf")
        print("2. Install chromium: sudo apt-get install chromium-browser")
        print("3. Open the HTML file in a browser and use Print > Save as PDF")
        print(f"\nHTML file is ready at: {html_file}")
        return False

    if os.path.exists(pdf_file):
        file_size = os.path.getsize(pdf_file) / 1024 / 1024  # MB
        print(f"\n🎉 PDF created successfully!")
        print(f"   Location: {pdf_file}")
        print(f"   Size: {file_size:.2f} MB")
        return True
    else:
        print(f"\n❌ PDF file was not created")
        return False

if __name__ == "__main__":
    convert_html_to_pdf()
