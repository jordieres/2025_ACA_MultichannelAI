from typing import List


class PromptBuilder:
    """Static class to build prompts for various LLM-based analysis tasks."""

    @staticmethod
    def prompt_qa(text: str) -> List[dict[str, str]]:
        """Builds a prompt to classify a meeting intervention as Question, Answer, or Procedure.

        Args:
            text (str): The intervention text to classify.

        Returns:
            List[dict[str, str]]: A list of messages formatted for an LLM chat input.
        """
        return [
            {
                'role': 'system',
                'content': """
                You are a model designed to classify interventions in meetings or conferences into three categories:
                [Question]: If the intervention has an interrogative tone or seeks information.
                [Answer]: If the intervention provides information or responds to a previous question.
                [Procedure]: If the intervention is part of the meeting protocol, such as acknowledgments, moderation steps, or phrases without substantial informational content.
                """
            },
            {
                'role': 'user',
                'content': f'Here is the text of the intervention: "{text}"'
            }
        ]

    @staticmethod
    def prompt_10k(text: str) -> List[dict[str, str]]:
        """Builds a prompt to classify a text into 10-K sections: Business, Risk Factors, MD&A, or Other.

        Args:
            text (str): The financial or business-related intervention.

        Returns:
            List[dict[str, str]]: A list of formatted chat messages for the LLM.
        """
        return [
            {
                'role': 'system',
                'content': """
                You are an expert in financial reporting and the structure of the SEC Form 10-K. Your task is to analyze a given text excerpt from an earnings call or conference 
                and classify it into one of the following sections: Business, Risk Factors, MD&A, or Other.

                [Business]: Describes the company’s core operations, market strategy, and structure.
                [Risk Factors]: Highlights regulatory, financial, or operational risks.
                [MD&A]: Management's Discussion and Analysis of financial performance.
                [Other]: If it doesn’t fit in any of the above.
                """
            },
            {
                'role': 'user',
                'content': f'Here is the text of the intervention: "{text}"'
            }
        ]

    @staticmethod
    def explain_why_other(text: str) -> List[dict[str, str]]:
        """Builds a prompt asking the LLM to explain why a given text was classified as 'Other'.

        Args:
            text (str): The text originally classified as Other.

        Returns:
            List[dict[str, str]]: A list of formatted chat messages for the LLM.
        """
        return [
            {
                'role': 'system',
                'content': """
                You are an expert in financial reporting and SEC Form 10-K structure.
                A previous intervention was classified as [Other].
                Analyze the intervention and explain why it doesn’t fit the Business, Risk Factors, MD&A, or Financial Statements categories.
                """
            },
            {
                'role': 'user',
                'content': f'Here is the text of the intervention: "{text}"'
            }
        ]

    @staticmethod
    def analize_qa(intervention: str, response: str) -> List[dict[str, str]]:
        """Builds a prompt to analyze whether a multi-question intervention was answered fully, partially, or not at all.

        Args:
            intervention (str): The original question or set of questions.
            response (str): The corresponding response from the speaker.

        Returns:
            List[dict[str, str]]: A list of messages formatted for the LLM to assess QA coverage.
        """
        return [
            {
                "role": "system",
                "content": """
                You are an assistant that analyzes multi-question interventions from financial conference calls.

                Given an intervention and a response:
                - Extract all questions.
                - Determine if each is answered (yes/partially/no).
                - If answered, include a summary and quote.

                Return the output in valid JSON format as specified.
                """
            },
            {
                "role": "user",
                "content": f"""
                Intervention:
                {intervention}

                Response:
                {response}
                """
            }
        ]

    @staticmethod
    def prompt_monologue(text: str) -> List[dict[str, str]]:
        """Builds a prompt to classify an intervention as a Monologue or Procedure.

        Args:
            text (str): The intervention text.

        Returns:
            List[dict[str, str]]: A list of formatted chat messages.
        """
        return [
            {
                'role': 'system',
                'content': """
                You are a model designed to classify interventions in meetings or conferences into:
                [Monologue]: Detailed statements by company executives containing insights, plans, or results.
                [Procedure]: Protocol steps like greetings, acknowledgments, or moderation without real content.
                """
            },
            {
                'role': 'user',
                'content': f'Here is the text of the intervention: "{text}"'
            }
        ]

    @staticmethod
    def check_coherence(monologue: str, response: str) -> List[dict[str, str]]:
        """Builds a prompt to evaluate the coherence between a monologue and a response.

        Args:
            monologue (str): The earlier speech or monologue from the executive.
            response (str): The follow-up response to be checked for logical consistency.

        Returns:
            List[dict[str, str]]: A list of messages formatted for the LLM to assess coherence.
        """
        return [
            {
                "role": "system",
                "content": """
                You are an expert in business communication and consistency analysis.

                Task:
                - Check if the response topic is present in the monologue.
                - Determine logical consistency.
                - If inconsistent, provide excerpts and explanations.

                Return the output as JSON:
                {
                  "topic_covered": true | false,
                  "consistent": true | false,
                  "summary": "...",
                  "contradictions": [
                    {
                      "monologue_excerpt": "...",
                      "response_excerpt": "...",
                      "explanation": "..."
                    }
                  ]
                }
                """
            },
            {
                "role": "user",
                "content": f"""
                Monologue:
                {monologue}

                Response:
                {response}
                """
            }
        ]
