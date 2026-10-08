import asyncio
from playwright.async_api import async_playwright

async def main():
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page()
        page.on('console', lambda msg: print(f'CONSOLE: {msg.type}: {msg.text}'))
        page.on('pageerror', lambda err: print(f'PAGE ERROR: {err}'))
        await page.goto('http://localhost:8000')
        print('Loaded page')
        await page.click('#navGallery')
        await page.wait_for_selector('.btn-open')
        print('Gallery loaded')
        await page.click('.btn-open')
        print('Opened model')
        await page.wait_for_timeout(2000)
        await page.click('#navAuth')
        await page.click('#tabRegister')
        await page.fill('#regUser', 'test2')
        await page.fill('#regEmail', 'test2@test.com')
        await page.fill('#regPass', 'pass')
        
        page.on('dialog', lambda dialog: print(f'DIALOG: {dialog.message}'))
        await page.click('#btnSubmitRegister')
        await page.wait_for_timeout(1000)
        await browser.close()

asyncio.run(main())
