class PromptBuilder:
    @staticmethod
    def prompt_qa(text: str) -> list:
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
                'content': f"""
                Here is the text of the intervention: "{text}"
                """
            }
        ]

    @staticmethod
    def prompt_10k(text: str) -> list:
        return [
            {
                'role': 'system',
                'content': 
                """
                You are an expert in financial reporting and the structure of the SEC Form 10-K. Your task is to analyze a given text excerpt from an earnings call or conference 
                where an S&P 500 company presents its financial and operational updates. You must classify it into one of the following sections of the 10-K report, 
                ensuring there is no overlap or ambiguity.

                [Business]: Describes the company's core operations, including its main products or services, target markets, competitive positioning, industry landscape, 
                corporate structure, and long-term strategy. Discussions about mergers, acquisitions, partnerships, and market expansions fall under this category. 
                Do not include financial performance, risk analysis, or management interpretations.

                [Risk Factors]: Identifies significant risks that could negatively impact the company, such as regulatory challenges, macroeconomic factors, 
                supply chain disruptions, cybersecurity threats, litigation, or market competition. This section is strictly focused on identifying potential risks, 
                not discussing actual financial performance or strategic decisions.

                [Selected Financial Data]: Provides a high-level summary of key financial metrics over past fiscal years, such as revenue, net income, earnings per share (EPS), 
                and return on assets (ROA). This section presents raw numerical data but does not provide explanations or insights into financial trends.

                [MD&A]: Contains management's discussion and analysis of financial results, explaining trends, operational performance, revenue drivers, cost structures, 
                and forward-looking expectations. This section provides context for financial data, identifying why certain numbers changed and how the company is responding to 
                economic conditions.

                [Financial Statements and Supplementary Data]: Includes official, audited financial statements such as the income statement, balance sheet, and cash flow statement.
                It also contains detailed accounting disclosures, footnotes, and regulatory reporting requirements. This section presents raw financial figures without 
                interpretation or strategic commentary.

                [Other]: If the text does not fit into any of the categories above, classify it here.
                """
            },
            {
                'role': 'user',
                'content': f"""    
                Here is the text of the intervention: "{text}"
                """
            }
        ]

    @staticmethod
    def explain_why_other(text: str) -> list:
        return [
            {
                'role': 'system',
                'content': 
                """
                You are an expert in financial reporting and the structure of the SEC Form 10-K. 
                Previously, a text excerpt was analyzed to classify it into one of the following categories:
                
                [Business]: Outlines the company's operations, including its main products/services, target markets, market presence, and strategic objectives.
                [Risk Factors]: Identifies significant risks (market, regulatory, financial) that could adversely affect the company's stability or performance.
                [Selected Financial Data]: Highlights key financial metrics (revenue, net income, assets) over recent years, offering a high-level financial snapshot.
                [MD&A]: Analyzes financial results, explaining trends, challenges, and strategies, and provides management's perspective on future performance.
                [Financial Statements and Supplementary Data]: Contains audited financial statements (income statement, balance sheet, cash flow) and detailed notes adhering to accounting standards. 
                
                The excerpt was classified as [Other], meaning it did not fit into any of the above categories. 

                Your task is to carefully analyze the text again and provide a detailed explanation of why it does not align with any of these categories. 
                """
            },
            {
                'role': 'user',
                'content': f"""
                Here is the text of the intervention: "{text}"
                """
            }
        ]