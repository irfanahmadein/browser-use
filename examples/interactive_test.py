
import os
import json
import asyncio
import logging
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from langchain_openai import ChatOpenAI
from browser_use import Agent
from browser_use.controller.service import Controller


class TestSession:
    def __init__(
            self,
            prompt: str,
            headless: bool = True,
            take_screenshots: bool = False,
            device: str = 'desktop',
            debug: bool = False
    ):
        # Set configuration attributes first
        self.prompt = prompt
        self.headless = headless
        self.take_screenshots = take_screenshots
        self.device = device
        self.debug = debug

        # Initialize results tracking
        self.steps: List[Dict] = []
        self.conversations: List[Dict] = []

        # Create directory structure
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.base_dir = Path('test_sessions')
        self.session_dir = self.base_dir / timestamp
        self.screenshots_dir = self.session_dir / 'screenshots'
        self.results_file = self.session_dir / 'results.json'
        self.conversation_file = self.session_dir / 'conversation.json'
        self.log_file = self.session_dir / 'test.log'

        # Create directories
        self._create_directories()

        # Setup logging after directories are created
        self._setup_logging()

        # Log initial configuration
        self.logger.info("=== Test Session Initialized ===")
        self.logger.info(f"Prompt: {self.prompt}")
        self.logger.info(f"Headless: {self.headless}")
        self.logger.info(f"Screenshots: {self.take_screenshots}")
        self.logger.info(f"Device: {self.device}")
        self.logger.info(f"Session Directory: {self.session_dir}")

    def _create_directories(self):
        """Create necessary directories"""
        try:
            # Create main session directory
            self.session_dir.mkdir(parents=True, exist_ok=True)

            # Create screenshots directory if screenshots are enabled
            if self.take_screenshots:
                self.screenshots_dir.mkdir(parents=True, exist_ok=True)

            print(f"Created session directory: {self.session_dir}")

        except Exception as e:
            print(f"Error creating directories: {str(e)}")
            raise

    def _setup_logging(self):
        """Setup logging configuration"""
        try:
            # Create formatter
            formatter = logging.Formatter(
                '%(asctime)s - %(levelname)s - %(message)s'
            )

            # Setup file handler
            file_handler = logging.FileHandler(self.log_file)
            file_handler.setFormatter(formatter)

            # Setup console handler
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)

            # Setup logger
            self.logger = logging.getLogger(f"TestSession_{self.session_dir.name}")
            self.logger.setLevel(logging.DEBUG if self.debug else logging.INFO)
            self.logger.addHandler(file_handler)
            self.logger.addHandler(console_handler)

        except Exception as e:
            print(f"Error setting up logging: {str(e)}")
            raise

    def add_step(self, step_number: int, action: str, status: str,
                 details: Optional[str] = None, screenshot_path: Optional[str] = None,
                 error: Optional[str] = None):
        """Add a step to the test execution history"""
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

    def add_conversation(self, step_number: int, role: str, content: str):
        """Add a conversation entry"""
        message = {
            'step_number': step_number,
            'timestamp': datetime.now().isoformat(),
            'role': role,
            'content': content
        }
        self.conversations.append(message)
        self.logger.debug(f"Conversation recorded: {message}")

    def save_results(self):
        """Save test results and conversations"""
        try:
            # Prepare results
            results = {
                'test_configuration': {
                    'prompt': self.prompt,
                    'headless': self.headless,
                    'screenshots_enabled': self.take_screenshots,
                    'device': self.device,
                    'debug': self.debug,
                    'timestamp': datetime.now().isoformat()
                },
                'steps': self.steps
            }

            # Save results
            with open(self.results_file, 'w') as f:
                json.dump(results, f, indent=2)
            self.logger.info(f"Results saved to: {self.results_file}")

            # Save conversations
            with open(self.conversation_file, 'w') as f:
                json.dump(self.conversations, f, indent=2)
            self.logger.info(f"Conversations saved to: {self.conversation_file}")

        except Exception as e:
            self.logger.error(f"Failed to save results: {str(e)}")
            raise

    async def run(self):
        """Execute the test session"""
        self.logger.info("=== Starting Test Execution ===")

        try:
            # Initialize controller
            controller = Controller(headless=self.headless)

            # Configure browser
            if self.take_screenshots:
                controller.browser.take_screenshots = True
                controller.browser.screenshots_dir = str(self.screenshots_dir)
            controller.browser.device = self.device

            # Initialize agent
            agent = Agent(
                task=self.prompt,
                llm=ChatOpenAI(model='gpt-4o'),
                controller=controller,
                use_vision=True
            )

            # Wrap the agent's execute step to track conversations
            original_step = agent.step

            async def step_with_logging():
                try:
                    step_number = len(self.steps) + 1
                    self.logger.info(f"\nExecuting Step {step_number}")

                    # Execute the step
                    await original_step()

                    # Get latest screenshot if available
                    screenshot_path = None
                    if self.take_screenshots:
                        screenshots = list(self.screenshots_dir.glob('*.png'))
                        if screenshots:
                            screenshot_path = max(screenshots, key=os.path.getctime)

                    # Record step results
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

            # Replace the step method
            agent.step = step_with_logging

            # Run the agent
            history = await agent.run()

            # Save final results
            self.save_results()

            self.logger.info("=== Test Execution Completed ===")
            return history

        except Exception as e:
            self.logger.error(f"Test execution failed: {str(e)}", exc_info=True)
            raise
        finally:
            self.save_results()


def main():
    parser = argparse.ArgumentParser(description='Interactive Browser Testing')
    parser.add_argument('--prompt', required=True, help='Task prompt for the agent')
    parser.add_argument('--no-headless', action='store_true', help='Run in visible mode')
    parser.add_argument('--screenshot', action='store_true', help='Enable screenshots')
    parser.add_argument('--device', choices=['desktop', 'mobile', 'tablet'],
                        default='desktop', help='Device viewport to simulate')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')

    args = parser.parse_args()

    asyncio.run(TestSession(
        prompt=args.prompt,
        headless=not args.no_headless,
        take_screenshots=args.screenshot,
        device=args.device,
        debug=args.debug
    ).run())


if __name__ == "__main__":
    main()
