import streamlit as st
import preprocessor
import helper
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import emoji
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import plotly.graph_objs as go

def analyze_sentiment(text):
    analyzer = SentimentIntensityAnalyzer()
    sentiment_score = analyzer.polarity_scores(text)
    compound_score = sentiment_score['compound']
    if compound_score >= 0.05:
        return 'positive'
    elif compound_score <= -0.05:
        return 'negative'
    else:
        return 'neutral'

def contains_emoji(text):
    return any(char in emoji.UNICODE_EMOJI_ENGLISH for char in text)

def extract_emojis(text):
    return ''.join(c for c in text if c in emoji.UNICODE_EMOJI_ENGLISH)

st.sidebar.title("WhatsApp Chat Analyzer")

uploaded_file = st.sidebar.file_uploader("Choose a file")
if uploaded_file is not None:
    bytes_data = uploaded_file.getvalue()
    data = bytes_data.decode("utf-8")
    df = preprocessor.preprocess(data)

    # Fetch unique users
    user_list = df['user'].unique().tolist()
    user_list.remove('group_notification')
    user_list.sort()
    user_list.insert(0, "Overall")

    selected_user = st.sidebar.selectbox("Show analysis wrt", user_list)

    if st.sidebar.button("Show Analysis"):
        # Add sentiment analysis to the DataFrame
        df['sentiment'] = df['message'].apply(analyze_sentiment)

        # Split data into with and without emojis
        df['contains_emoji'] = df['message'].apply(contains_emoji)
        df_with_emoji = df[df['contains_emoji']]
        df_without_emoji = df[~df['contains_emoji']]

        # Visualize sentiment analysis with and without emojis
        col1, col2 = st.columns(2)

        with col1:
            st.title("Sentiment Analysis with Emojis")
            if df_with_emoji.empty:
                st.write("No messages with emojis to display.")
            else:
                sentiment_counts_emoji = df_with_emoji['sentiment'].value_counts()
                fig, ax = plt.subplots()
                ax.pie(sentiment_counts_emoji, labels=sentiment_counts_emoji.index, autopct='%1.1f%%', startangle=90)
                ax.axis('equal')
                st.pyplot(fig)

        with col2:
            st.title("Sentiment Analysis without Emojis")
            if df_without_emoji.empty:
                st.write("No messages without emojis to display.")
            else:
                sentiment_counts_no_emoji = df_without_emoji['sentiment'].value_counts()
                fig, ax = plt.subplots()
                ax.pie(sentiment_counts_no_emoji, labels=sentiment_counts_no_emoji.index, autopct='%1.1f%%', startangle=90)
                ax.axis('equal')
                st.pyplot(fig)

        # Bar chart of sentiment over time
        st.title("Sentiment Over Time")
        sentiment_over_time = df.groupby(['only_date', 'sentiment']).size().unstack(fill_value=0)

        # Filter dates to display only a subset
        date_subset = sentiment_over_time.index[::7]  # Display every 7th date
        sentiment_over_time_subset = sentiment_over_time.loc[date_subset]

        sentiment_over_time_subset.plot(kind='bar', stacked=True, figsize=(10, 6))
        plt.xlabel('Date')
        plt.ylabel('Count')
        plt.title('Sentiment Over Time')
        plt.xticks(rotation='vertical')  # Rotate x-axis labels for better readability
        st.pyplot(plt)

        if selected_user == 'Overall':
            st.title('Sentiment Analysis by User')
            pivot = pd.pivot_table(df, index='sentiment', columns='user', values='message', aggfunc='count').apply(
                lambda x: x / x.sum(), axis=0)
            heatmap = go.Figure(data=go.Heatmap(
                z=pivot.values,
                x=pivot.columns,
                y=pivot.index,
                hovertemplate='Sentiment: %{y}<br>User: %{x}<br>Proportion: %{z:.2%}<extra></extra>',
                colorscale='Greens'))
            heatmap.update_layout(
                width=900, height=600,
                title='Proportion of Sentiments by User',
                xaxis={'type': 'category'},
                yaxis=dict(title='Sentiment Types'),
                autosize=False
            )
            st.plotly_chart(heatmap, use_container_width=True)

    if st.sidebar.button("Show Stats"):
        # Stats Area
        num_messages, words, num_media_messages, num_links = helper.fetch_stats(selected_user, df)
        st.title("Top Statistics")
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.header("Total Messages")
            st.title(num_messages)
        with col2:
            st.header("Total Words")
            st.title(words)
        with col3:
            st.header("Media Shared")
            st.title(num_media_messages)
        with col4:
            st.header("Links Shared")
            st.title(num_links)

        # monthly timeline
        st.title("Monthly Timeline")
        timeline = helper.monthly_timeline(selected_user, df)
        fig, ax = plt.subplots()
        ax.plot(timeline['time'], timeline['message'], color='green')
        plt.xticks(rotation='vertical')
        st.pyplot(fig)

        # daily timeline
        st.title("Daily Timeline")
        daily_timeline = helper.daily_timeline(selected_user, df)
        fig, ax = plt.subplots()
        ax.plot(daily_timeline['only_date'], daily_timeline['message'], color='black')
        plt.xticks(rotation='vertical')
        st.pyplot(fig)

        # activity map
        st.title('Activity Map')
        col1, col2 = st.columns(2)

        with col1:
            st.header("Most busy day")
            busy_day = helper.week_activity_map(selected_user, df)
            fig, ax = plt.subplots()
            ax.bar(busy_day.index, busy_day.values, color='purple')
            plt.xticks(rotation='vertical')
            st.pyplot(fig)

        with col2:
            st.header("Most busy month")
            busy_month = helper.month_activity_map(selected_user, df)
            fig, ax = plt.subplots()
            ax.bar(busy_month.index, busy_month.values, color='orange')
            plt.xticks(rotation='vertical')
            st.pyplot(fig)

        st.title("Weekly Activity Map")
        user_heatmap = helper.activity_heatmap(selected_user, df)
        fig, ax = plt.subplots()
        ax = sns.heatmap(user_heatmap)
        st.pyplot(fig)

        # finding the busiest users in the group(Group level)
        if selected_user == 'Overall':
            st.title('Most Busy Users')
            x, new_df = helper.most_busy_users(df)
            fig, ax = plt.subplots()
            col1, col2 = st.columns(2)

            with col1:
                ax.bar(x.index, x.values, color='red')
                plt.xticks(rotation='vertical')
                st.pyplot(fig)
            with col2:
                st.dataframe(new_df)

        # WordCloud
        st.title("Wordcloud")
        df_wc = helper.create_wordcloud(selected_user, df)
        fig, ax = plt.subplots()
        ax.imshow(df_wc)
        st.pyplot(fig)

        # most common words
        most_common_df = helper.most_common_words(selected_user, df)

        fig, ax = plt.subplots()

        ax.barh(most_common_df[0], most_common_df[1])
        plt.xticks(rotation='vertical')

        st.title('Most common words')
        st.pyplot(fig)

        # emoji analysis
        emoji_df = helper.emoji_helper(selected_user, df)
        st.title("Emoji Analysis")

        col1, col2 = st.columns(2)

        with col1:
            st.dataframe(emoji_df)
        with col2:
            fig, ax = plt.subplots()
            ax.pie(emoji_df[1].head(), labels=emoji_df[0].head(), autopct="%0.2f")
            st.pyplot(fig)