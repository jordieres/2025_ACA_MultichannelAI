import pandas as pd
from typing import Optional, List
from tqdm import tqdm

from multimodal_fin.analysis_qa_effects.pipelines.company_pipeline import CompanyPipeline
from multimodal_fin.analysis_qa_effects.config import PipelineConfig

class MultiCompanyRunner:
    def __init__(self, pipeline: CompanyPipeline, config: PipelineConfig, use_tqdm: bool = False):
        self.pipeline = pipeline
        self.config = config
        self.use_tqdm = use_tqdm

    def run(self, companies: List[str], plot_company: Optional[str] = None) -> pd.DataFrame:
        all_results = []

        iterator = (
            tqdm(companies, desc="Processing companies")
            if self.use_tqdm
            else companies
        )

        for company in iterator:
            try:
                df_tests = self.pipeline.run(company, plot_if_company=plot_company)
                for _, r in df_tests.iterrows():
                    for modality in ["audio", "text"]:
                        all_results.append({
                            "company": company,
                            "emotion": r["emotion"],
                            "modality": modality,
                            "hedges_g": r[f"{modality}_hedges_g"],
                            "ci_low": r[f"{modality}_g_ci_low"],
                            "ci_high": r[f"{modality}_g_ci_high"],
                            "p_value": r[f"{modality}_p_value"],
                        })
            except Exception as e:
                msg = f"❌ Skipping {company} due to unexpected error: {e}"
                if self.use_tqdm:
                    tqdm.write(msg)
                else:
                    print(msg)
                continue

        return pd.DataFrame(all_results)