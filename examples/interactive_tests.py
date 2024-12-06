import os
import json
import asyncio
import logging
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from browser_use import Agent
from browser_use.browser.browser import Browser, BrowserConfig
from browser_use.browser.context import BrowserContextConfig
from browser_use.controller.service import Controller


class TestSession:
    def __init__(
            self,
            prompt: str,
            headless: bool = False,
            take_screenshots: bool = True,
            device: str = 'desktop',
            debug: bool = False,
            provider: str = 'openai'
    ):
        # Initialize all basic attributes first
        self.prompt = prompt
        self.device = device
        self.debug = debug
        self.provider = provider

        # Create directory structure
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.base_dir = Path('test_sessions')
        self.session_dir = self.base_dir / timestamp
        self.screenshots_dir = self.session_dir / 'screenshots'
        self.results_file = self.session_dir / 'results.json'
        self.conversation_file = self.session_dir / 'conversation.json'
        self.log_file = self.session_dir / 'test.log'

        # Initialize browser config before creating directories
        self.browser_config = BrowserConfig(
            headless=headless,
            take_screenshots=take_screenshots,
            screenshots_dir=str(self.screenshots_dir) if take_screenshots else None,
            new_context_config=BrowserContextConfig(
                take_screenshots=take_screenshots,
                screenshots_dir=str(self.screenshots_dir) if take_screenshots else None
            )
        )

        self._create_directories()
        self._setup_logging()

        # Initialize results tracking
        self.steps = []
        self.conversations = []

        self.logger.info("=== Test Session Initialized ===")
        self.logger.info(f"Prompt: {self.prompt}")
        self.logger.info(f"Provider: {self.provider}")
        self.logger.info(f"Device: {self.device}")
        self.logger.info(f"Session Directory: {self.session_dir}")

    def _create_directories(self):
        self.session_dir.mkdir(parents=True, exist_ok=True)
        if self.browser_config.take_screenshots:
            self.screenshots_dir.mkdir(parents=True, exist_ok=True)

    def _setup_logging(self):
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

        file_handler = logging.FileHandler(self.log_file)
        file_handler.setFormatter(formatter)

        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)

        self.logger = logging.getLogger(f"TestSession_{self.session_dir.name}")
        self.logger.setLevel(logging.DEBUG if self.debug else logging.INFO)
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)

    def get_llm(self):
        if self.provider == 'anthropic':
            return ChatAnthropic(
                model_name='claude-3-5-sonnet-20240620',
                timeout=25,
                stop=None,
                temperature=0.0
            )
        return ChatOpenAI(model='gpt-4o', temperature=0.0)

    async def run(self):
        try:
            browser = Browser(config=self.browser_config)

            async with await browser.new_context(
                    config=BrowserContextConfig(
                        take_screenshots=True,
                        screenshots_dir=str(self.screenshots_dir),
                        disable_security=False
                    )
            ) as browser_context:
                controller = Controller()
                controller.browser = browser

                agent = Agent(
                    task=self.prompt,
                    llm=self.get_llm(),
                    controller=controller,
                    browser_context=browser_context,
                    use_vision=True
                )

                original_step = agent.step

                async def step_with_logging():
                    try:
                        step_number = len(self.steps) + 1
                        self.logger.info(f"\nExecuting Step {step_number}")
                        await original_step()

                        screenshot_path = None
                        if os.path.exists(self.screenshots_dir):
                            screenshots = list(self.screenshots_dir.glob('*.png'))
                            if screenshots:
                                screenshot_path = max(screenshots, key=os.path.getctime)
                                self.logger.info(f"Found screenshot: {screenshot_path}")

                        self.add_step(
                            step_number=step_number,
                            action=f"Step {step_number}",
                            status="completed",
                            screenshot_path=screenshot_path
                        )

                    except Exception as e:
                        self.logger.error(f"Step {step_number} failed: {str(e)}")
                        self.add_step(
                            step_number=step_number,
                            action=f"Step {step_number}",
                            status="error",
                            error=str(e)
                        )
                        raise

                agent.step = step_with_logging
                history = await agent.run()

            await browser.close()
            self.save_results()
            return history

        except Exception as e:
            self.logger.error(f"Test execution failed: {str(e)}", exc_info=True)
            raise
        finally:
            self.save_results()

    def add_step(self, step_number: int, action: str, status: str,
                 details: Optional[str] = None, screenshot_path: Optional[str] = None,
                 error: Optional[str] = None):
        step = {
            'step_number': step_number,
            'timestamp': datetime.now().isoformat(),
            'action': action,
            'status': status,
            'details': details,
            'screenshot_path': str(screenshot_path) if screenshot_path else None,
            'error': error
        }
        self.steps.append(step)
        self.logger.debug(f"Step recorded: {step}")

    def save_results(self):
        try:
            results = {
                'test_configuration': {
                    'prompt': self.prompt,
                    'provider': self.provider,
                    'device': self.device,
                    'debug': self.debug,
                    'timestamp': datetime.now().isoformat()
                },
                'steps': self.steps
            }

            with open(self.results_file, 'w') as f:
                json.dump(results, f, indent=2)
            self.logger.info(f"Results saved to: {self.results_file}")

            with open(self.conversation_file, 'w') as f:
                json.dump(self.conversations, f, indent=2)
            self.logger.info(f"Conversations saved to: {self.conversation_file}")

        except Exception as e:
            self.logger.error(f"Failed to save results: {str(e)}")
            raise


def main():
    parser = argparse.ArgumentParser(description='Interactive Browser Testing')
    parser.add_argument('--prompt', required=True, help='Task prompt for the agent')
    parser.add_argument('--no-headless', action='store_true', help='Run in visible mode')
    parser.add_argument('--screenshot', action='store_true', help='Enable screenshots')
    parser.add_argument('--device', choices=['desktop', 'mobile', 'tablet'],
                        default='desktop', help='Device viewport to simulate')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    parser.add_argument('--provider', choices=['openai', 'anthropic'],
                        default='openai', help='The model provider to use')

    args = parser.parse_args()

    asyncio.run(TestSession(
        prompt=args.prompt,
        headless=not args.no_headless,
        take_screenshots=args.screenshot,
        device=args.device,
        debug=args.debug,
        provider=args.provider
    ).run())


if __name__ == "__main__":
    main()