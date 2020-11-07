#!/usr/bin/env python
"""
High-level exatomic widget test script. Start up
a jupyter notebook server, use selenium to interact
with the browser, and execute widget rendering.
Optionally persist a reference PNG of the rendered
widget, e.g. to compare to a reference PNG for
strict validation.


Configure the script by environment variables:

    VENDOR_TYPE = chrome OR firefox (default chrome)
    HEADLESS_RUN = true OR false (default true)
    CLEANUP_POST = true OR false (default true)

    python test_widget.py


Or by command line (takes priority over environment,
specify as few as necessary):

    python test_widget.py \
        --Selenium.vendor_type=firefox|chrome \
        --Selenium.browser_path=/usr/bin/google-chrome-stable \
        --Selenium.headless_run=true|false \
        --Selenium.console_port='9222' \
        --Selenium.driver_timeout=10 \
        --Notebook.server_url='http://path/to/server' \
        --Notebook.server_port='8889' \
        --Notebook.server_token='mystatictoken' \
        --Notebook.cleanup_post=true|false
"""

import os
import sys
import logging.config

from uuid import uuid4
from time import sleep
from shutil import rmtree, which
from subprocess import Popen, PIPE
from importlib import import_module

from traitlets.config.application import Application
from traitlets.config.configurable import Configurable
from traitlets import Int, Unicode, Bool, default, validate

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.wait import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.action_chains import ActionChains


logging.basicConfig()
logging.getLogger('test_widget').setLevel(logging.INFO)
DEFAULT_VENDOR_TYPE = 'chrome'
DEFAULT_HEADLESS_RUN = 'true'
DEFAULT_CLEANUP_POST = 'true'
DEFAULT_CONSOLE_PORT = '9222'
DEFAULT_DRIVER_TIMEOUT = 10


class Base:

    @property
    def log(self):
        log = logging.getLogger(
            '.'.join(['test_widget', self.__class__.__name__]))
        return log

    def clean_bool(self, val):
        self.log.debug(f'cleaning {type(val)} {val} to bool')
        if isinstance(val, bool):
            return val
        return {'true': True, 'false': False}[val.lower()]


class Selenium(Base, Configurable):
    """A selenium interface for custom widgets in the jupyter notebook."""
    vendor_type = Unicode('').tag(config=True)
    browser_path = Unicode('').tag(config=True)
    headless_run = Bool(False).tag(config=True)
    console_port = Unicode(DEFAULT_CONSOLE_PORT).tag(config=True)
    driver_timeout = Int(DEFAULT_DRIVER_TIMEOUT).tag(config=True)

    @default('vendor_type')
    def _default_vendor_type(self):
        val = os.getenv('VENDOR_TYPE', DEFAULT_VENDOR_TYPE)
        self.log.debug(f'_default_vendor_type: {val}')
        return val

    @default('browser_path')
    def _default_browser_path(self):
        aliases = {
            'chrome': ['google-chrome-stable', 'google-chrome'],
        }.get(self.vendor_type, [self.vendor_type])
        path = None
        for alias in aliases:
            try:
                path = which(alias)
                self.log.info(f'found browser path: {path}')
                break
            except Exception:
                continue
        if path is None:
            raise Exception(f'browser executable not found for vendor_type={self.vendor_type}')
        return path

    @validate('headless_run')
    def _validate_headless_run(self, prop):
        prop['value'] = self.clean_bool(prop['value'])
        return prop['value']

    @default('headless_run')
    def _default_headless_run(self):
        val = os.getenv('HEADLESS_RUN', DEFAULT_HEADLESS_RUN).lower()
        return self.clean_bool(val)

    @staticmethod
    def click_by_css(driver, wait, css):
        wait.until(
            EC.presence_of_element_located((By.CSS_SELECTOR, css))
        )
        [obj] = driver.find_elements_by_css_selector(css)
        obj.click()

    def get_vendor_options(self):
        mod = f'selenium.webdriver.{self.vendor_type}.options'
        self.log.info(f'importing {mod}')
        mod = import_module(mod)
        options = mod.Options()

        # parametrize or apply vendor-specific config
        if self.vendor_type == 'firefox':
            self.log.info('no firefox-specific options')
            headless_arg = '-headless'
            binary_attr = 'binary'

        elif self.vendor_type == 'chrome':
            headless_arg = '--headless'
            binary_attr = 'binary_location'
            self.log.info('adding chrome-specific options')
            options.add_argument('--no-sandbox')
            options.add_argument('--disable-gpu')
            options.add_argument('--disable-extensions')
            options.add_argument('--disable-dev-shm-usage')
            options.add_argument(f'--remote-debugging-port={self.console_port}')

        else:
            self.log.info('unfamiliar vendor_type {self.vendor_type}')
            self.log.info('guessing chrome-like driver options')
            headless_arg = '--headless'
            binary_attr = 'binary_location'

        # apply parametrized config
        self.log.info(f'setting options.{binary_attr} to {self.browser_path}')
        setattr(options, binary_attr, self.browser_path)
        self.log.info(f'adding {headless_arg} flag {self.headless_run}')
        if self.headless_run:
            options.add_argument(headless_arg)
        return options

    def init_webdriver(self, scratch):
        log_path = f'{scratch}/{self.vendor_type}driver.log'
        proper = self.vendor_type.title()
        cls = getattr(webdriver, proper, None)
        if cls is None:
            raise Exception(f'no webdriver {proper} found')
        return cls(options=self.get_vendor_options(), service_log_path=log_path)

    def create_and_goto_new_notebook(self, driver, wait):
        # click on new notebook dropdown
        self.log.info('opening new notebook')
        self.click_by_css(driver, wait, '#new-dropdown-button')
        # primary notebook server window
        n_initial = len(driver.window_handles)
        # spawns new notebook
        self.click_by_css(driver, wait, '#kernel-python3 > a')
        # wait for new notebook
        wait.until(
            lambda driver: n_initial != len(driver.window_handles)
        )
        # switch to new notebook
        self.log.info('navigating to new notebook')
        driver.switch_to.window(driver.window_handles[1])

    def insert_and_execute_first_cell(self, driver, wait, code_to_run):
        # select the first input cell
        self.log.info('selecting first cell')
        cell = '#notebook-container > div > div.input > div.inner_cell > div.input_area'
        self.click_by_css(driver, wait, cell)
        # input some text
        self.log.info(f"writing '{code_to_run}'")
        ActionChains(driver).send_keys(code_to_run).perform()
        [cell] = driver.find_elements_by_css_selector(cell)
        # execute the cell
        self.log.info('executing first cell')
        (ActionChains(driver).key_down(Keys.SHIFT)
                             .key_down(Keys.ENTER)
                             .key_up(Keys.SHIFT)
                             .key_up(Keys.ENTER)
                             .perform())

    def close_and_leave_new_notebook(self, driver, wait):
        # shut down the new notebook
        self.log.info('clicking on file menu button')
        menu = '#filelink'
        self.click_by_css(driver, wait, menu)

        self.log.info('clicking on close and halt')
        n_current = len(driver.window_handles)
        close = '#close_and_halt > a'
        self.click_by_css(driver, wait, close)

        # TODO : convention to handle browser-specific logic
        if self.vendor_type == 'chrome':
            # closing triggers "Leave without saving? alert"
            wait.until(EC.alert_is_present())
            driver.switch_to.alert.accept()
            self.log.info('caught alert about unsaved notebook')

        # wait until second notebook window is closed
        wait.until(
            lambda driver: n_current != len(driver.window_handles)
        )

        # switch back to server home window
        driver.switch_to.window(driver.window_handles[0])

    def shutdown_notebook_server(self, driver, wait):
        self.log.info('closing notebook server down from UI')
        self.click_by_css(driver, wait, '#shutdown')

    def run_basic(self, server_url, scratch_dir):

        with self.init_webdriver(scratch_dir) as driver:

            wait = WebDriverWait(driver, self.driver_timeout)
            driver.get(server_url)

            # spawn a notebook and render a widget
            self.create_and_goto_new_notebook(driver, wait)
            code = 'import exatomic; exatomic.widgets.widget_base.Scene()'
            self.insert_and_execute_first_cell(driver, wait, code)

            # touch the rendered canvas
            self.log.info('clicking on the widget')
            scene = ('#notebook-container > '
                     'div.cell.code_cell.rendered.unselected > '
                     'div.output_wrapper > div.output > div > '
                     'div.output_subarea.jupyter-widgets-view > '
                     'div > canvas')
            self.click_by_css(driver, wait, scene)

            # download a PNG of the widget
            self.log.info('screenshotting the widget')
            driver.get_screenshot_as_file(f'{scratch_dir}/widget.png')

            # shut it down
            self.close_and_leave_new_notebook(driver, wait)
            self.shutdown_notebook_server(driver, wait)
            driver.close()


class Notebook(Base, Configurable):
    server_url = Unicode('').tag(config=True)
    server_port = Unicode('8889').tag(config=True)
    server_token = Unicode(str(uuid4())).tag(config=True)
    cleanup_post = Bool(True).tag(config=True)

    @default('server_url')
    def _default_server_url(self):
        return f'http://localhost:{self.server_port}/?token={self.server_token}'

    @validate('cleanup_post')
    def _validate_cleanup_post(self, prop):
        prop['value'] = self.clean_bool(prop['value'])
        return prop['value']

    @default('cleanup_post')
    def _default_cleanup_post(self):
        val = os.getenv('CLEANUP_POST', DEFAULT_CLEANUP_POST).lower()
        return self.clean_bool(val)

    def start_notebook_server(self):
        self.log.info(f'making scratch dir {self.server_token}')
        os.makedirs(self.server_token, exist_ok=True)
        self.log.info(f'starting notebook server {self.server_url}')
        command = [
            'jupyter', 'notebook', '--no-browser',
            f'--NotebookApp.port={self.server_port}',
            f'--NotebookApp.token={self.server_token}'
        ]
        self.log.info(f"full command {' '.join(command)}")
        self.server_process = Popen(
                command, stdout=PIPE, stderr=PIPE, cwd=self.server_token)

    def stop_notebook_server(self):
        proc = getattr(self, 'server_process')
        if proc is not None:
            self.log.info('stopping notebook server')
            proc.kill()
        if self.cleanup_post and os.path.isdir(self.server_token):
            rmtree(self.server_token)


class App(Base, Application):

    def _initialize(self):
        self.notebook = Notebook()
        self.selenium = Selenium()

    def initialize(self, argv=None):
        self.parse_command_line(argv)
        self.notebook = Notebook(config=self.config)
        self.selenium = Selenium(config=self.config)
        self.log.info(f'custom config: {self.config}')

    def run(self):
        nb = self.notebook
        nb.start_notebook_server()
        try:
            sleep(2)
            self.selenium.run_basic(nb.server_url, nb.server_token)
        finally:
            nb.stop_notebook_server()


def test_selenium():
    """pytest flags seems to interfere with the
    traitlets command-line system so use a work
    around for now.
    """
    app = App()
    app._initialize()
    app.run()

def startup_info():
    log = logging.getLogger('test_widget')
    width = 12
    spacer = f'{{:<{width}}}{{:>{width}}}'.format
    header = spacer('env', 'val')
    log.info(header)
    log.info('-' * len(header))
    for env in [
            'VENDOR_TYPE',
            'HEADLESS_RUN',
            'CLEANUP_POST',
            'MOZ_HEADLESS',
    ]:
        log.info(spacer(env, os.getenv(env, 'N/A')))


if __name__ == '__main__':
    startup_info()
    app = App()
    app.initialize(sys.argv)
    app.run()
