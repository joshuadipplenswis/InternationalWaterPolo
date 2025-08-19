import os
os.environ["STREAMLIT_DISABLE_WATCHDOG_WARNING"] = "true"
os.environ["WATCHDOG_DISABLE_FILE_WATCHING"] = "true"
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind
import openpyxl
from pathlib import Path  # ✅ correct import

st.set_page_config(layout="wide")

def read_excel_table(file, sheet_name: str, table_name: str):
    try:
        wb = openpyxl.load_workbook(file, data_only=True)
        sheet = wb[sheet_name]
        table = sheet.tables[table_name]
        ref = table.ref

        # Parse the cell range from the reference
        min_col, min_row, max_col, max_row = openpyxl.utils.range_boundaries(ref)

        # Read the values from the sheet
        data = []
        for row in sheet.iter_rows(min_row=min_row, max_row=max_row,
                                   min_col=min_col, max_col=max_col,
                                   values_only=True):
            data.append(row)

        # Convert to DataFrame
        df = pd.DataFrame(data[1:], columns=data[0])

        # Drop unnamed columns
        df = df.loc[:, ~df.columns.astype(str).str.startswith('Unnamed')]

        return df

    except Exception as e:
        st.error(f"❌ Error reading table '{table_name}' from sheet '{sheet_name}': {e}")
        return None


def cohen_d(x, y):
    nx = len(x)
    ny = len(y)
    pooled_std = np.sqrt(((nx - 1) * np.std(x, ddof=1) ** 2 +
                          (ny - 1) * np.std(y, ddof=1) ** 2) /
                         (nx + ny - 2))
    return (np.mean(x) - np.mean(y)) / pooled_std if pooled_std != 0 else 0


def main():
    st.title("📊 Water Polo International Analysis Page")

    from pathlib import Path

    # Always resolve relative to the app directory
    DATA_PATH = Path(__file__).parent / "Winning_Losing_Teams.xlsx"
    st.write(f"Looking for file at: {DATA_PATH.resolve()}")  # Debug: shows the full absolute path

    df_win = read_excel_table(DATA_PATH, "Winning Teams", "Table1")
    df_loss = read_excel_table(DATA_PATH, "Losing Teams", "Table2")

    if df_win is None or df_loss is None:
        st.error("❌ Could not read data from the Excel file.")
        st.stop()

        # 👇 Competition filter
        if 'Competition' in df_win.columns and 'Competition' in df_loss.columns:
            st.markdown("### 🔍 Filter by Competition")
            competition_options = sorted(set(df_win['Competition'].dropna()) | set(df_loss['Competition'].dropna()))
            selected_competitions = st.multiselect(
                "Select Competitions to Include",
                competition_options,
                default=competition_options
            )
            if selected_competitions:
                df_win = df_win[df_win['Competition'].isin(selected_competitions)]
                df_loss = df_loss[df_loss['Competition'].isin(selected_competitions)]
        else:
            st.warning("⚠️ 'Competition' column not found in one or both sheets.")

        try:
            # Drop non-numeric for analysis and align columns
            num_cols = df_win.select_dtypes(include=np.number).columns.intersection(df_loss.select_dtypes(include=np.number).columns)

            import plotly.express as px

            # Global Summary
            avg_win = df_win[num_cols].mean()
            avg_loss = df_loss[num_cols].mean()

            global_df = pd.DataFrame({'Winning Teams': avg_win, 'Losing Teams': avg_loss})
            global_df = global_df.sort_index()

            # Reset and melt for Plotly format
            global_df_reset = global_df.reset_index().rename(columns={'index': 'Statistic'})
            global_df_melted = global_df_reset.melt(id_vars='Statistic', var_name='Result', value_name='Average Value')

            st.subheader("📌 Global Average Comparison (Winning vs Losing Teams)")

            fig = px.bar(
                global_df_melted,
                y='Statistic',
                x='Average Value',
                color='Result',
                barmode='group',
                orientation='h',
                hover_name='Statistic',
                hover_data={'Average Value': ':.2f', 'Result': True},
                color_discrete_map={'Winning Teams': '#66c2a5', 'Losing Teams': '#fc8d62'},
                height=800
            )

            fig.update_layout(xaxis_title="Average Value", yaxis_title="Statistic")
            st.plotly_chart(fig, use_container_width=True)

            # Cohen's d
            d_values = {col: cohen_d(df_win[col].dropna(), df_loss[col].dropna()) for col in num_cols}
            d_df = pd.DataFrame.from_dict(d_values, orient='index', columns=['Cohen_d']).sort_values('Cohen_d', key=abs)

            st.subheader("📌 Cohen's d of Global Differences")

            # ➕ Explanation for coaches
            st.markdown("""
            **ℹ️ What is Cohen’s d?**  
            Cohen’s d measures how different winning and losing teams are for each stat.  
            - **0.2 = small difference**  
            - **0.5 = moderate difference**  
            - **0.8+ = large difference**  
            The bigger the number, the more that stat separates wins from losses.
            """)

            fig, ax = plt.subplots(figsize=(10, 8))
            sns.barplot(x='Cohen_d', y=d_df.index, data=d_df, ax=ax)
            ax.set_xlabel("Cohen's d Value")
            ax.set_ylabel("Statistic")
            st.pyplot(fig)

            # Create a dictionary of Cohen's d values from common numeric stats
            try:
                common_stats = list(set(df_win.columns) & set(df_loss.columns))

                # Filter for strictly numeric columns only (and exclude timedelta columns)
                valid_numeric_cols = df_win[common_stats].select_dtypes(include=[np.number]).columns
                ignore_cols = ['Days Since Latest', 'Time Weight']  # Add any others you calculate manually
                common_stats = [col for col in valid_numeric_cols if col not in ignore_cols]

                cohen_d_values = {}
                for stat in common_stats:
                    x = df_win[stat].dropna()
                    y = df_loss[stat].dropna()
                    if len(x) > 1 and len(y) > 1:
                        cohen_d_values[stat] = cohen_d(x, y)

                cohen_d_df = pd.DataFrame.from_dict(cohen_d_values, orient='index', columns=['Cohen_d'])
            except Exception as e:
                st.error(f"Error computing Cohen's d values: {e}")
                cohen_d_df = pd.DataFrame()  # fallback to empty if error

            st.subheader("📌 Performance Index")

            st.markdown("""
            **ℹ️ What contributes to the Performance Index?**

            The Performance Index is a weighted combination of:
            - **Win Rate** (50% weight)
            - **Average Goal Margin**: Goals Scored minus Goals Conceded (20% weight)
            - **Key Match Statistics**: Weighted by Cohen’s d values, showing which stats most distinguish wins from losses (30% weight)
            """)

            # 🔝 Calculate and visualize weighted team performance index (no 'Goal Difference' required)
            with st.expander("🔝 Team Rankings by Performance Index", expanded=True):
                try:
                    # Merge winning and losing data
                    df_win['Result'] = 'Win'
                    df_loss['Result'] = 'Lose'
                    df_combined = pd.concat([df_win, df_loss], ignore_index=True)

                    # --- Time Decay Weighting ---
                    df_combined['Date'] = pd.to_datetime(df_combined['Date'], errors='coerce')
                    df_combined = df_combined.dropna(subset=['Date'])  # Ensure no null dates
                    latest_date = df_combined['Date'].max()
                    df_combined['Days Since Latest'] = (latest_date - df_combined['Date']).dt.days
                    decay_rate = 0.005  # You can tweak this for faster/slower decay
                    df_combined['Time Weight'] = np.exp(-decay_rate * df_combined['Days Since Latest'])

                    # ✅ Step 1: Compute Cohen's d
                    numeric_stats = df_combined.select_dtypes(include=[np.number]).columns.tolist()
                    ignore_cols = ['Days Since Latest', 'Time Weight']
                    numeric_stats = [col for col in numeric_stats if col not in ignore_cols]

                    cohen_d_values = {}
                    for stat in numeric_stats:
                        x = df_win[stat].dropna()
                        y = df_loss[stat].dropna()
                        if len(x) > 1 and len(y) > 1:
                            pooled_std = np.sqrt(
                                ((len(x) - 1) * x.std(ddof=1) ** 2 + (len(y) - 1) * y.std(ddof=1) ** 2) / (
                                            len(x) + len(y) - 2)
                            )
                            d = (x.mean() - y.mean()) / pooled_std if pooled_std != 0 else 0
                            cohen_d_values[stat] = abs(d)

                    cohen_d_df = pd.DataFrame.from_dict(cohen_d_values, orient='index', columns=['Cohen_d'])

                    # ✅ Step 2: Calculate team-level stats with time weights
                    team_stats = []
                    all_teams = pd.unique(df_combined[['Winning Team', 'Losing Team']].values.ravel('K'))
                    all_teams = [team for team in all_teams if pd.notna(team)]

                    for team in all_teams:
                        team_games = df_combined[
                            (df_combined['Winning Team'] == team) | (df_combined['Losing Team'] == team)
                            ].copy()

                        if team_games.empty:
                            continue

                        # Win Rate
                        wins = team_games[team_games['Winning Team'] == team]
                        losses = team_games[team_games['Losing Team'] == team]
                        total_games = len(wins) + len(losses)
                        win_rate = len(wins) / total_games if total_games else 0

                        # Goal Margin
                        margin_list = []
                        for _, row in wins.iterrows():
                            if pd.notna(row.get('Goals Scored')) and pd.notna(row.get('Goals Conceded')):
                                margin_list.append(row['Goals Scored'] - row['Goals Conceded'])
                        for _, row in losses.iterrows():
                            if pd.notna(row.get('Goals Scored')) and pd.notna(row.get('Goals Conceded')):
                                margin_list.append(row['Goals Scored'] - row['Goals Conceded'])
                        avg_margin = np.mean(margin_list) if margin_list else 0

                        # Weighted average of all relevant stats using time decay
                        relevant_stats = list(cohen_d_df.index)
                        team_stat_df = team_games.copy()
                        team_stat_df['Is Team'] = (
                                (team_stat_df['Winning Team'] == team) | (team_stat_df['Losing Team'] == team)
                        )
                        team_stat_df = team_stat_df[team_stat_df['Is Team']]
                        team_stat_df = team_stat_df[relevant_stats + ['Time Weight']].dropna(subset=relevant_stats)

                        if not team_stat_df.empty:
                            weighted_stats = (
                                team_stat_df[relevant_stats].multiply(team_stat_df['Time Weight'], axis=0)).sum()
                            total_weight = team_stat_df['Time Weight'].sum()
                            weighted_avg_stats = (weighted_stats / total_weight) if total_weight > 0 else pd.Series(0,
                                                                                                                    index=relevant_stats)
                        else:
                            weighted_avg_stats = pd.Series(0, index=relevant_stats)

                        team_data = {
                            'Team': team,
                            'Win Rate': win_rate,
                            'Goal Margin': avg_margin,
                        }
                        team_data.update(weighted_avg_stats)
                        team_stats.append(team_data)

                    df_perf = pd.DataFrame(team_stats).set_index('Team').fillna(0)

                    # ✅ Step 3: Stat Score using Cohen's d
                    # Define stats where lower values are better
                    lower_is_better_stats = {
                        "Goals Conceded",
                        "Penalties Conceded",
                        "Missed Shots",
                        "Shots Blocked",
                        "SoT Conceded",
                        "Exclusions Conceded",
                        "Pass to CF Conceded",
                        "Pass to CF - Outcome"
                    }

                    # Adjust stats so that all are "higher is better" by inverting where needed
                    adjusted_stats = df_perf[relevant_stats].copy()

                    for stat in relevant_stats:
                        if stat in lower_is_better_stats:
                            max_val = df_perf[stat].max()
                            adjusted_stats[stat] = max_val - df_perf[stat]  # invert the scale

                    # Recalculate the stat score with adjusted values
                    df_perf['Stat Score'] = adjusted_stats.mul(cohen_d_df['Cohen_d'], axis=1).sum(axis=1)

                    # ✅ Step 4: Final Performance Index
                    df_perf['Performance Index'] = (
                            df_perf['Win Rate'] * 0.5 +
                            (df_perf['Goal Margin'] / df_perf['Goal Margin'].abs().max() if df_perf[
                                                                                                'Goal Margin'].abs().max() != 0 else 0) * 0.2 +
                            (df_perf['Stat Score'] / df_perf['Stat Score'].max() if df_perf[
                                                                                        'Stat Score'].max() != 0 else 0) * 0.3
                    )

                    # ✅ Step 5: Plotly Chart
                    import plotly.express as px
                    sorted_perf = df_perf.sort_values("Performance Index", ascending=False).reset_index()

                    fig = px.bar(
                        sorted_perf,
                        x="Performance Index",
                        y="Team",
                        orientation='h',
                        text="Performance Index",
                        title="🔝 Team Performance Index Ranking (Time-Weighted)",
                        color="Performance Index",
                        color_continuous_scale="Blues"
                    )
                    fig.update_layout(
                        yaxis=dict(autorange="reversed"),
                        xaxis_title="Performance Index",
                        yaxis_title="Team",
                        margin=dict(l=100, r=20, t=50, b=40),
                        height=600
                    )
                    fig.update_traces(texttemplate='%{text:.2f}', textposition='outside')
                    st.plotly_chart(fig, use_container_width=True)

                except Exception as e:
                    st.error(f"Error calculating performance index: {e}")

            # ⬇️ Team Statistic Comparison Table

            st.subheader("📌 Team Statistical Comparison Table")

            with st.expander("🆚 Team vs Team Stat Comparison Table", expanded=True):
                try:
                    # Merge win/loss data into one DataFrame
                    df_combined_all = pd.concat([
                        df_win.rename(columns={'Winning Team': 'Team'}),
                        df_loss.rename(columns={'Losing Team': 'Team'})
                    ], ignore_index=True)

                    # Keep only numeric columns
                    numeric_cols = df_combined_all.select_dtypes(include=[np.number]).columns

                    # Compute average stats per team
                    team_avg_stats = df_combined_all.groupby('Team')[numeric_cols].mean().dropna(how='all')
                    teams = sorted(team_avg_stats.index.unique())

                    # Team selection
                    teams = sorted(team_avg_stats.index.unique())

                    team1 = st.selectbox("Select Team 1", teams, key="team1_compare_table")
                    team2 = st.selectbox("Select Team 2", teams, key="team2_compare_table")

                    if team1 not in team_avg_stats.index or team2 not in team_avg_stats.index:
                        st.warning("One or both selected teams do not have enough data to display a comparison.")
                        st.stop()

                    # Rankings
                    # Define stats where lower is better
                    lower_is_better_stats = {
                        "Goals Conceded",
                        "Penalties Conceded",
                        "Missed Shots",
                        "Shots Blocked",
                        "SoT Conceded",
                        "Exclusions Conceded",
                        "Pass to CF Conceded",
                        "Pass to CF - Outcome"
                    }

                    # Create rankings DataFrame with stat-specific ordering
                    rankings = pd.DataFrame(index=team_avg_stats.index)

                    for stat in team_avg_stats.columns:
                        if stat in lower_is_better_stats:
                            rankings[stat] = team_avg_stats[stat].rank(ascending=True)  # lower is better
                        else:
                            rankings[stat] = team_avg_stats[stat].rank(ascending=False)  # higher is better

                    # Build side-by-side comparison table
                    df_compare = pd.DataFrame({
                        f"{team1} Avg": team_avg_stats.loc[team1],
                        f"{team1} Rank": rankings.loc[team1],
                        "Statistic": team_avg_stats.columns,
                        f"{team2} Avg": team_avg_stats.loc[team2],
                        f"{team2} Rank": rankings.loc[team2],
                    })

                    # Reorder columns: [Team1 Avg, Team1 Rank, Statistic, Team2 Avg, Team2 Rank]
                    df_compare = df_compare[
                        [f"{team1} Avg", f"{team1} Rank", "Statistic", f"{team2} Avg", f"{team2} Rank"]
                    ].set_index("Statistic")

                    # Format ranking columns as integers
                    def ordinal(n):
                        n = int(n)
                        return f"{n}{'tsnrhtdd'[(n // 10 % 10 != 1) * (n % 10 < 4) * n % 10::4]}"

                    df_compare[f"{team1} Rank"] = df_compare[f"{team1} Rank"].apply(
                        lambda x: ordinal(x) if pd.notnull(x) else "")
                    df_compare[f"{team2} Rank"] = df_compare[f"{team2} Rank"].apply(
                        lambda x: ordinal(x) if pd.notnull(x) else "")

                    # Color only top 3 and bottom 3 ranks
                    def highlight_top_bottom(val):
                        if isinstance(val, str) and val[:-2].isdigit():
                            rank = int(val[:-2])  # remove 'st', 'nd', etc.
                            if rank <= 3:
                                return 'background-color: rgba(102, 194, 165, 0.8); font-weight: bold'  # soft green
                            elif rank >= 10:
                                return 'background-color: rgba(252, 141, 98, 0.8); font-weight: bold'  # soft red
                        return ''

                    # Apply style to both team rank columns
                    styled_df = df_compare.style.applymap(highlight_top_bottom,
                                                          subset=[f"{team1} Rank", f"{team2} Rank"]) \
                        .format(precision=2)

                    st.dataframe(styled_df, use_container_width=True)

                except Exception as e:
                    st.error(f"❌ Error generating comparison table: {e}")

            # Team Filter
            team_col_win = "Winning Team"
            team_col_loss = "Losing Team"

            team_names = sorted(set(df_win[team_col_win].dropna().unique()) | set(df_loss[team_col_loss].dropna().unique()))
            selected_team = st.selectbox("🔍 Select a Team for Breakdown", team_names)

            team_win_data = df_win[df_win[team_col_win] == selected_team]
            team_loss_data = df_loss[df_loss[team_col_loss] == selected_team]

            # --- Snapshot Stats ---
            total_matches = len(team_win_data) + len(team_loss_data)
            total_wins = len(team_win_data)
            total_losses = len(team_loss_data)
            win_percentage = (total_wins / total_matches * 100) if total_matches > 0 else 0

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Matches Played", total_matches)
            col2.metric("Wins", total_wins)
            col3.metric("Losses", total_losses)
            col4.metric("Win %", f"{win_percentage:.1f}%")

            if not team_win_data.empty and not team_loss_data.empty:
                avg_team_win = team_win_data[num_cols].mean()
                avg_team_loss = team_loss_data[num_cols].mean()

                team_df = pd.DataFrame({'Wins': avg_team_win, 'Losses': avg_team_loss})
                team_df = team_df.sort_index()

                import plotly.express as px

                st.subheader(f"📌 {selected_team} - Average Stats in Wins vs Losses")

                # Reset index so 'Statistic' is a column
                team_df_reset = team_df.reset_index().rename(columns={'index': 'Statistic'})

                # Melt the dataframe so we have one column for win/loss category and one for values
                team_df_melted = team_df_reset.melt(id_vars='Statistic', var_name='Result', value_name='Average Value')

                # Create interactive Plotly bar chart with soft red/green
                fig = px.bar(
                    team_df_melted,
                    y='Statistic',
                    x='Average Value',
                    color='Result',
                    barmode='group',
                    orientation='h',
                    hover_name='Statistic',
                    hover_data={'Average Value': ':.2f', 'Result': True},
                    color_discrete_map={
                        'Wins': '#66c2a5',  # soft green
                        'Losses': '#fc8d62'  # soft coral/red
                    },
                    height=800
                )

                fig.update_layout(xaxis_title="Average Value", yaxis_title="Statistic")
                st.plotly_chart(fig, use_container_width=True)

                st.subheader(f"📌 {selected_team} - Boxplots of All Stats in Wins vs Losses")
                combined = pd.concat([
                    team_win_data[num_cols].assign(Result='Win'),
                    team_loss_data[num_cols].assign(Result='Loss')
                ])
                melted = pd.melt(combined, id_vars='Result', var_name='Statistic', value_name='Value')

                import plotly.express as px

                fig = px.box(
                    melted,
                    x="Statistic",
                    y="Value",
                    color="Result",
                    points="all",  # Show individual data points
                    hover_data={"Value": True, "Statistic": True, "Result": True},
                    title=f"{selected_team} - Boxplots of All Stats in Wins vs Losses",
                    color_discrete_map={
                        "Win": "green",
                        "Loss": "red"
                    }
                )

                fig.update_layout(
                    xaxis_title="Statistic",
                    yaxis_title="Value",
                    boxmode="group",
                    height=800,  # Increased height from 600 to 800
                    margin=dict(l=40, r=40, t=60, b=60)
                )

                fig.update_traces(marker=dict(size=5, opacity=0.6))  # Optional: fine-tune point visibility

                st.plotly_chart(fig, use_container_width=True)

            else:
                st.warning(f"No data found for {selected_team} in one or both result categories.")

        except Exception as e:
            st.error(f"❌ Error loading Excel tables: {e}")

        # --- 🔍 Team Summary Insight ---

        try:
            st.subheader(f"🧠 Insights: What matters when {selected_team} win or lose")

            # Filter team-specific data for wins and losses
            team_win_data = df_win[df_win['Winning Team'] == selected_team]
            team_loss_data = df_loss[df_loss['Losing Team'] == selected_team]

            # Only include numeric stats
            common_stats = list(set(team_win_data.columns) & set(team_loss_data.columns))
            valid_cols = team_win_data[common_stats].select_dtypes(include=[np.number]).columns
            ignore_cols = [
                'Days Since Latest', 'Time Weight',
                'Goals Scored', 'Goals Conceded'  # <- Exclude obvious outcome metrics
            ]
            stat_cols = [col for col in valid_cols if col not in ignore_cols]

            # Compute Cohen's d
            team_cohen_d = {}
            for stat in stat_cols:
                win_vals = team_win_data[stat].dropna()
                loss_vals = team_loss_data[stat].dropna()
                if len(win_vals) > 1 and len(loss_vals) > 1:
                    d = cohen_d(win_vals, loss_vals)
                    team_cohen_d[stat] = d

            # Sort by absolute effect size and get top 5
            top_stats = sorted(team_cohen_d.items(), key=lambda x: abs(x[1]), reverse=True)[:5]

            # Build summary strings
            insights = []
            for stat, d in top_stats:
                avg_win = team_win_data[stat].mean()
                avg_loss = team_loss_data[stat].mean()
                better_when = "higher" if avg_win > avg_loss else "lower"
                outcome = "increased" if better_when == "higher" else "reduced"
                insights.append(
                    f"- When {selected_team} **win**, they tend to have **{outcome}** `{stat}` compared to when they lose.")

            st.markdown("### 📋 Summary:")
            for insight in insights:
                st.markdown(insight)

        except Exception as e:
            st.error(f"❌ Error generating win/loss insight summary: {e}")

if __name__ == "__main__":
    main()