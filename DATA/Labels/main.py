import pandas as pd
from event import Event

# Carga del DataFrame de conferencias (ejemplo)
# conf_df = pd.read_csv('datos_conferencias.csv')  # columnas: timestamp, symbol
df_conferences = pd.read_csv('/home/aacastro/Alejandro/ACA_MultichanelAI_2025/2025_ACA_MultichannelAI/DATA/Labels/conference_data.csv', index_col=0)    
# conf_df = df_conferences.iloc[[1, 15]]
# conf_df = df_conferences.head(5)
# conf_df = df_conferences[df_conferences['symbol'] == 'AMZN']
# conf_df

event_summaries = []

for _, row in df_conferences.iterrows():
    try:
        event = Event(
            event_date=row['timestamp'],
            ticker=row['symbol'],
            year=str(row['year']),
            quarter=row['quarter'],
            plot_base_path='plots_eventos'
        )
        event.generate_and_save_plots()
        event_summaries.append(event.summary_dict())
    except Exception as e:
        print(f"❌ Error con {row['symbol']} en {row['timestamp']}: {e}")

df_summary = pd.DataFrame(event_summaries)
df_summary.to_csv("event_summaries.csv", index=False)
print("✅ Gráficos y resumen guardados correctamente.")
