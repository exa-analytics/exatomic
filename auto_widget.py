#!/usr/bin/env python
"""

Assumes the following environment:

    export BROWSER=/path/to/chrome/executable
    export NOTEBOOK_TOKEN=notebook_token_hash

    pip install selenium

And that a chromedriver version matching BROWSER
is in the PATH. (See install-chromedriver.sh)

Note:
    Hit enter to pass a failure to load extension
    and continue the test. Not sure why disable-extensions
    does not suppress that pop-up.

"""
import os

from selenium import webdriver

from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.wait import WebDriverWait

from selenium.webdriver.chrome.options import Options


def click_by_css(driver, wait, css):
    wait.until(
        EC.presence_of_element_located((By.CSS_SELECTOR, css))
    )
    [obj] = driver.find_elements_by_css_selector(css)
    obj.click()


def run_notebook_widget():
    options = Options()

    options.binary_location = os.getenv('BROWSER')
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-extensions')
    options.add_argument('--disable-dev-shm-usage')

    token = os.getenv('NOTEBOOK_TOKEN')
    notebook = f'http://localhost:8888/?token={token}'

    with webdriver.Chrome(options=options) as driver:
        wait = WebDriverWait(driver, 10)
        browser = driver.get(notebook)

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

        # the rendered canvas
        scene = ('#notebook-container > div.cell.code_cell.rendered.unselected > div.output_wrapper '
                 '> div.output > div > div.output_subarea.jupyter-widgets-view > div > canvas')
        click_by_css(driver, wait, scene)

        print('interacted with scene')

        # try to shut it down
        menu = '#filelink'
        click_by_css(driver, wait, menu)
        print('clicked on file menu button')

        close = '#close_and_halt > a'
        click_by_css(driver, wait, menu)
        print('clicked on close and halt')

        # may trigger "Leave without saving? alert"
        wait.until(EC.alert_is_present())
        driver.switch_to.alert.accept()
        print("why it not close?!")


if __name__ == '__main__':
    print("""\
Assumes the following environment:

    export BROWSER=/path/to/chrome/executable
    export NOTEBOOK_TOKEN=notebook_token_hash

    pip install selenium""")
    run_notebook_widget()
