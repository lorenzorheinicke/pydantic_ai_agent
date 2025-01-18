Get Transcript
This API fetches transcript/subtitles from YouTube videos in various formats and languages.

Quick Start
Request

curl -X GET 'https://api.supadata.ai/v1/youtube/transcript?url=https://www.youtube.com/watch?v=dQw4w9WgXcQ&text=true' \
 -H 'x-api-key: YOUR_API_KEY'
Response

{
"content": "Never gonna give you up, never gonna let you down...",
"lang": "en",
"availableLangs": ["en", "es", "zh-TW"]
}
Specification
Endpoint
GET https://api.supadata.ai/v1/youtube/transcript

Each request requires an x-api-key header with your API key available after signing up. Find out more about Authentication.

Query Parameters
Parameter Type Required Description
url string Yes* YouTube video URL. See Supported YouTube URL Formats.
videoId string Yes* YouTube video ID. Alternative to URL
lang string No Preferred language code of the transcript (ISO 639-1). See Languages.
text boolean No When true, returns plain text transcript. Default: false
chunkSize number No Maximum characters per transcript chunk (only when text=false)

- Either url or videoId must be provided

Response Format
When text=true:

{
"content": string,
"lang": string // ISO 639-1 language code
"availableLangs": string[] // List of available languages
}
When text=false:

{
"content": [
{
"text": string, // Transcript segment
"offset": number, // Start time in milliseconds
"duration": number, // Duration in milliseconds
"lang": string // ISO 639-1 language code of chunk
}
],
"lang": string // ISO 639-1 language code of transcript
"availableLangs": string[] // List of available languages
}
Error Codes
The API returns HTTP status codes and error codes. See this page for more details.

Supported YouTube URL Formats
url parameter supports various YouTube URL formats. See this page for more details.

Languages
The endpoint supports multiple languages. The lang parameter is used to specify the preferred language of the transcript. If the video does not have a transcript in the preferred language, the endpoint will return a transcript in the first available language and a list of other available languages. It is then possible to make another request to get the transcript in your chosen fallback language.

Need to get your transcript in a language not yet supported? Check the Transcript Translation endpoint.
