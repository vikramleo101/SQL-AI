import google.generativeai as genai
import os

# Make sure your API key is set as an environment variable
# or configured directly with genai.configure(api_key="YOUR_API_KEY")
genai.configure(api_key="AIzaSyDOoYv3z4ukixqlpq99CN22gMf4XBstiNo")

print("Available models that support generateContent:")
for m in genai.list_models():
  if 'generateContent' in m.supported_generation_methods:
    print(m.name)