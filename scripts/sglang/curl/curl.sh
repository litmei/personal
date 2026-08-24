curl --location 'http://127.0.0.1:8880/generate' \
    --header 'Content-Type: application/json' \
    --data '{
        "text": "介绍下秦始皇",
        "sampling_params": {
            "temperature": 0,
            "max_new_tokens": 20
        }
    }'
