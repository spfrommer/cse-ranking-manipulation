global="--run-eval"
# global=""

python natural.py --original $global --target-model mixtral-8x22
python natural.py --original $global --target-model gpt-3.5
python natural.py --original $global --target-model llama3-70b
python natural.py --original $global --target-model gpt-4-turbo
# Needs web server running
# python natural.py --original $global --target-model llama3-sonar-large-online --use-hosted-urls

python natural.py --rewritten $global --target-model mixtral-8x22
python natural.py --rewritten $global --target-model gpt-3.5
python natural.py --rewritten $global --target-model llama3-70b
python natural.py --rewritten $global --target-model gpt-4-turbo

python adversarial.py $global --target-model mixtral-8x22
python adversarial.py $global --target-model gpt-3.5
python adversarial.py $global --target-model llama3-70b
python adversarial.py $global --target-model gpt-4-turbo
# Needs web server running
# python adversarial.py $global --target-model llama3-sonar-large-online --use-hosted-urls --transfer-attacks-from-model gpt-4-turbo