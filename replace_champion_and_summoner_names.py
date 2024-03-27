import json


def format_dataframe(dataframe):
    # Make a copy of the original dataframe
    df_formatted = dataframe.copy()

    # Load champion names for team 1 from JSON file
    with open('dataset/champion_info.json') as f:
        champion_info_t1 = json.load(f)

    # Load champion names for team 2 from JSON file
    with open('dataset/champion_info.json') as f:
        champion_info_t2 = json.load(f)

    # Load summoner spell names from JSON file
    with open('dataset/summoner_spell_info.json') as f:
        summoner_spell_info = json.load(f)

    # Extract champion name mapping from champion_info
    champion_mapping_t1 = {str(champion['id']): champion['name'] for champion in champion_info_t1['data'].values()}
    champion_mapping_t2 = {str(champion['id']): champion['name'] for champion in champion_info_t2['data'].values()}

    # Extract summoner spell name mapping from summoner_spell_info
    summoner_spell_mapping = {spell_id: spell_data['name'] for spell_id, spell_data in summoner_spell_info['data'].items()}

    # Replace champion IDs with champion names in the DataFrame
    for col in df_formatted.columns:
        if 't1' in col and 'Kills' not in col:
            df_formatted[col] = df_formatted[col].astype(str).map(champion_mapping_t1)
        elif 't2' in col and 'Kills' not in col:
            df_formatted[col] = df_formatted[col].astype(str).map(champion_mapping_t2)
        elif 'sum' in col:
            df_formatted[col] = df_formatted[col].astype(str).map(summoner_spell_mapping)

    return df_formatted
