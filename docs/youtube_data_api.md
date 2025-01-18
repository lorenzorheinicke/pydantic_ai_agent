# Code example

import requests

def get_youtube_video_info(video_id, api_key): # Define the API endpoint
url = f"https://www.googleapis.com/youtube/v3/videos?id={video_id}&key={api_key}&part=snippet,statistics"

    # Make the API request
    response = requests.get(url)

    # Check if the request was successful
    if response.status_code == 200:
        data = response.json()

        # Check if we received any items
        if "items" in data and len(data["items"]) > 0:
            video_info = data["items"][0]
            title = video_info["snippet"]["title"]
            thumbnails = video_info["snippet"]["thumbnails"]
            statistics = video_info["statistics"]

            # Print out the information
            print(f"Title: {title}")
            print("Thumbnails:")
            for size, thumbnail in thumbnails.items():
                print(f"  {size.capitalize()}: {thumbnail['url']}")
            print("Statistics:")
            print(f"  View Count: {statistics.get('viewCount', 'N/A')}")
            print(f"  Like Count: {statistics.get('likeCount', 'N/A')}")
            print(f"  Dislike Count: {statistics.get('dislikeCount', 'N/A')}")
            print(f"  Comment Count: {statistics.get('commentCount', 'N/A')}")
        else:
            print("No video found with this ID.")
    else:
        print(f"Error: {response.status_code} - {response.text}")

# Replace with your actual video ID and API key

video_id = "YOUR_VIDEO_ID"
api_key = "YOUR_API_KEY"

get_youtube_video_info(video_id, api_key)
