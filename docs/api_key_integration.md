# API Key Integration and Cost Monitoring – 2025-03-05

## Overview

This document outlines the implementation of secure API key integration and cost monitoring for embedding generation using the OpenAI API. The integration ensures that all production and test environments use real API calls while providing safeguards against unexpected costs through a token usage estimation and warning system.

## API Key Management

### Implementation Details

1. **Secure Loading of API Key**:
   - The API key is loaded from environment variables using the python-dotenv library
   - A .env.example file is provided as a template for users to create their own .env file
   - The actual .env file is excluded from version control via .gitignore

2. **Error Handling**:
   - If no API key is found and mock mode is not explicitly enabled, a clear error message is raised
   - For backward compatibility, tests can still use mock mode by explicitly setting `use_mock=True`
   - The system logs the detection of the API key (showing only the first few characters for security)

## OpenAI API Version Compatibility

### API Structure Changes

1. **OpenAI Package Version**:
   - The implementation is compatible with OpenAI package version 0.28.1
   - This version requires using `openai.Embedding.create` instead of `openai.embeddings.create`
   - Future OpenAI SDK updates may require additional adjustments

2. **API Call Implementation**:
   ```python
   # Logging the API call
   logging.info(f"Generating embeddings for {len(texts)} chunks using OpenAI API")
   
   # Correct API structure for version 0.28.1
   response = openai.Embedding.create(
       model=self.embedding_config.model_name,
       input=texts,
       dimensions=self.embedding_config.dimensions
   )
   
   # Response parsing adjusted for the structure in 0.28.1
   for i, chunk in enumerate(chunks_to_embed):
       chunk.embedding = np.array(response['data'][i]['embedding'])
   ```

3. **Response Structure Handling**:
   - The response format in 0.28.1 has embeddings at `response['data'][i]['embedding']`
   - All code and tests have been updated to handle this structure correctly
   - Test mocks simulate this exact response structure for consistency

## Cost Monitoring System

### Token Usage Estimation

1. **Estimation Algorithm**:
   - A simple heuristic of approximately 0.25 tokens per character is used
   - This provides a reasonable estimate for most English text
   - The algorithm accounts for all text chunks being processed in a batch

2. **Implementation**:
   ```python
   def estimate_token_usage(text_list):
       """Estimate the number of tokens for a list of texts"""
       if not text_list:
           return 0
           
       # Simple estimation based on character count
       total_chars = sum(len(text) for text in text_list)
       estimated_tokens = int(total_chars * ESTIMATED_TOKENS_PER_CHAR)
       
       return estimated_tokens
   ```

### Warning System

1. **Threshold Configuration**:
   - Default threshold is set to 10,000 tokens
   - Users can override this in their .env file or through the EmbeddingConfig parameter
   - The threshold is checked before each API call

2. **Warning Messages**:
   - When the estimated token usage exceeds the threshold, a warning is logged
   - The warning includes the estimated token count for transparency
   - Both logging and console output are used to ensure visibility

## Changes to Embedding Generation

### Default Behavior Changes

1. **Mock Mode Changes**:
   - Mock mode is now explicitly disabled by default (`use_mock=False`)
   - Mock mode must be explicitly enabled when needed for testing
   - Automatic fallback to mock mode has been removed

2. **API Call Requirements**:
   - An API key is now required for embedding generation with OpenAI models
   - A clear ValueError is raised if the API key is missing
   - No automatic fallback to mock embeddings without explicit configuration

### Error Handling

1. **API Call Errors**:
   - Errors during API calls are no longer silently caught with fallback to mock
   - All API errors are propagated to the caller with clear error messages
   - This ensures users are aware of any issues with their API keys or requests

## Test Suite Adjustments

1. **New Tests**:
   - `test_api_key_requirement`: Verifies that an error is raised when no API key is available
   - `test_token_usage_estimation`: Validates the token estimation algorithm
   - `test_cost_warning_threshold`: Ensures warnings are triggered when thresholds are exceeded
   - `test_real_api_embedding`: Confirms that real API calls are made with the proper parameters

2. **Updated Tests**:
   - `test_generate_embeddings_with_mock_mode`: Modified to explicitly enable mock mode
   - All tests now respect the new API key requirements
   - Test mocks updated to reflect the correct API response structure for version 0.28.1

## Usage Example

```python
# Configure embedding parameters with cost monitoring
embedding_config = EmbeddingConfig(
    model_name="text-embedding-3-large",
    token_threshold=5000  # Custom threshold for cost warnings
)

# Initialize the chunking processor
chunking_processor = ChunkingProcessor(
    chunking_config=chunking_config,
    embedding_config=embedding_config
)

# Process segments into chunks with embeddings
# Will automatically log warnings if token usage exceeds threshold
chunks = chunking_processor.process_segments(segments)
```

## Testing and Verification

All tests have been updated and verified to:
1. Use the real API when an API key is available
2. Display appropriate warnings when token usage is high
3. Fail with clear error messages when an API key is missing
4. Honor the explicit mock mode setting when testing without API calls
5. Work with the OpenAI API version 0.28.1 structure

## Future Improvements

1. **Token Estimation Refinement**:
   - Implement a more sophisticated token counting algorithm that matches OpenAI's tokenizer
   - Add support for different languages in the token estimation

2. **Quota Management**:
   - Add ability to track total API usage across multiple calls
   - Implement optional daily/weekly quota limits

3. **Cost Tracking**:
   - Estimate and log the actual cost in dollars based on the OpenAI pricing
   - Aggregate usage statistics over time for budgeting 