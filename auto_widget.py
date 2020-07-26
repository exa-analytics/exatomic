#!/usr/bin/env python
"""

Assumes the following environment:

    export BROWSER=/path/to/chrome/executable

    pip install selenium

And that a chromedriver version matching BROWSER
is in the PATH. (See install-chromedriver.sh)

Note:
    Hit enter to pass a failure to load extension
    and continue the test. Not sure why disable-extensions
    does not suppress that pop-up.

"""
import os
import shutil
from uuid import uuid4
from time import sleep
import subprocess as sp

from selenium import webdriver
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.wait import WebDriverWait
from selenium.webdriver.chrome.options import Options


PORT = 8889
CONSOLE_PORT = 9222
TOKEN = str(uuid4())
TIMEOUT = 30


def start_notebook_server():
    os.makedirs(TOKEN, exist_ok=True)
    return sp.Popen(['jupyter', 'notebook', '--no-browser',
                     f'--NotebookApp.port={PORT}',
                     f'--NotebookApp.token={TOKEN}'],
                     stdout=sp.PIPE, stderr=sp.PIPE,
                     cwd=TOKEN)


def click_by_css(driver, wait, css):
    wait.until(
        EC.presence_of_element_located((By.CSS_SELECTOR, css))
    )
    [obj] = driver.find_elements_by_css_selector(css)
    obj.click()


def run_notebook_widget():
    options = Options()

    options.binary_location = os.getenv('BROWSER', 'usr/bin/google-chrome-stable')
    options.headless = True
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-gpu')
    options.add_argument('--disable-extensions')
    options.add_argument('--disable-dev-shm-usage')
    options.add_argument(f'--remote-debugging-port={CONSOLE_PORT}')

    notebook = f'http://localhost:{PORT}/?token={TOKEN}'
    print(notebook)

    with webdriver.Chrome(options=options) as driver:
        wait = WebDriverWait(driver, TIMEOUT)
        driver.get(notebook)

        # click on new notebook dropdown
        click_by_css(driver, wait, '#new-dropdown-button')
        # primary notebook server window
        initial_handles = driver.window_handles
        # spawns new notebook
        click_by_css(driver, wait, '#kernel-python3 > a')

        # wait for new notebook
        wait.until(
            lambda driver: len(initial_handles) != len(driver.window_handles)
        )
        # switch to new notebook
        driver.switch_to.window(driver.window_handles[1])

        # select the first input cell
        cell = '#notebook-container > div > div.input > div.inner_cell > div.input_area'
        click_by_css(driver, wait, cell)

        # input some text
        run = 'import exatomic; exatomic.widgets.widget_base.Scene()'
        ActionChains(driver).send_keys(run).perform()
        [cell] = driver.find_elements_by_css_selector(cell)

        # execute the cell
        (ActionChains(driver).key_down(Keys.SHIFT).key_down(Keys.ENTER)
                             .key_up(Keys.SHIFT).key_up(Keys.ENTER).perform())
        print('should show widget now')

        # touch the rendered canvas
        scene = ('#notebook-container > div.cell.code_cell.rendered.unselected > div.output_wrapper '
                 '> div.output > div > div.output_subarea.jupyter-widgets-view > div > canvas')
        click_by_css(driver, wait, scene)

        driver.get_screenshot_as_file(f'{TOKEN}/widget.png')

        print('interacted with scene')

        # shut down the new notebook
        menu = '#filelink'
        click_by_css(driver, wait, menu)
        print('clicked on file menu button')

        current_handles = driver.window_handles
        close = '#close_and_halt > a'
        click_by_css(driver, wait, close)
        print('clicked on close and halt')

        # may trigger "Leave without saving? alert"
        wait.until(EC.alert_is_present())
        driver.switch_to.alert.accept()
        print('caught alert about unsaved notebook')

        # wait until second notebook window is closed
        wait.until(
            lambda driver: len(current_handles) != len(driver.window_handles)
        )

        # switch back to server home window
        driver.switch_to.window(driver.window_handles[0])
        shutdown = '#shutdown'
        print('closing notebook server down from UI')
        click_by_css(driver, wait, shutdown)
        driver.close()
        print('closed server home window')


if __name__ == '__main__':
    print("""\
Assumes the following environment:

    export BROWSER=/path/to/chrome/executable

    pip install selenium
    
May get a "Failed to load extension" pop-up when
starting chrome via selenium. By hitting enter,
the test will continue and should run to completion.
""")
    server = start_notebook_server()
    try:
        sleep(0.2)
        run_notebook_widget()
    finally:
        server.kill()
        shutil.rmtree(TOKEN)
