# API Key Setup Guide for AI Chat Functionality

## Problem Identified
The AI chat is not answering user questions because **LLM API keys are not configured**. The system requires at least one LLM provider to generate responses.

## Required API Keys

### 1. OpenAI API Key
**Purpose**: Primary LLM provider for chat responses and embeddings
**Where to get it**: https://platform.openai.com/api-keys

**Steps to get OpenAI API key:**
1. Go to https://platform.openai.com
2. Sign up or log in to your account
3. Navigate to "API Keys" in the dashboard
4. Click "Create new secret key"
5. Copy the key and add it to your `.env` file

**Cost**: Pay-per-use (check OpenAI pricing)

### 2. Mistral API Key (Optional - for fallback)
**Purpose**: Alternative LLM provider if OpenAI is unavailable
**Where to get it**: https://console.mistral.ai/api-keys/

**Steps to get Mistral API key:**
1. Go to https://console.mistral.ai
2. Sign up or log in to your account
3. Navigate to "API Keys" section
4. Create a new API key
5. Copy the key and add it to your `.env` file

**Cost**: Pay-per-use (check Mistral pricing)

## Configuration Steps

### Step 1: Update Your .env File
Replace the placeholder values in your `.env` file:

```env
# OpenAI Configuration
OPENAI_API_KEY=sk-your-actual-openai-key-here
OPENAI_EMBEDDING_MODEL=text-embedding-3-large

# Mistral Configuration  
MISTRAL_API_KEY=your-actual-mistral-key-here
```

### Step 2: Restart the Backend Server
After updating the `.env` file, restart your backend server:

```bash
cd backend
python -m uvicorn app.main:app --reload
```

### Step 3: Test the Configuration
The system will automatically:
- Use OpenAI if both keys are available (preferred)
- Fall back to Mistral if OpenAI fails
- Show error if no API keys are configured

## System Behavior

### With Both API Keys Configured:
- ✅ **Primary**: Uses OpenAI GPT-4o-mini for chat responses
- ✅ **Fallback**: Uses Mistral if OpenAI fails
- ✅ **Embeddings**: Uses OpenAI text-embedding-3-large

### With Only One API Key:
- ✅ Uses the available provider
- ✅ Embeddings use the same provider

### With No API Keys:
- ❌ Chat functionality disabled
- ❌ Error: "LLM API keys not configured"

## Troubleshooting

### Common Issues:

1. **Invalid API Key**
   - Error: "Invalid API key"
   - Solution: Verify the key is copied correctly without extra spaces

2. **Rate Limiting**
   - Error: "Rate limit exceeded"
   - Solution: Wait or upgrade your API plan

3. **Insufficient Credits**
   - Error: "Insufficient credits"
   - Solution: Add billing information to your account

4. **Network Issues**
   - Error: "Network error" or timeout
   - Solution: Check internet connection and firewall settings

## Testing the Chat

Once configured, test the chat by:

1. Go to the chat page in your application
2. Select processed documents
3. Ask questions about the document content
4. The AI should respond with relevant answers based on the document content

## Security Notes

- Never commit actual API keys to version control
- Use environment variables for production
- Monitor your API usage to avoid unexpected charges
- Consider setting usage limits in your provider dashboard

## Support

If you encounter issues:
1. Check the backend logs for detailed error messages
2. Verify API keys are correctly set in the environment
3. Test API connectivity using the provider's playground
4. Check your account balance and billing status