import azure.functions as func
import datetime
import json
import logging

import os
from dotenv import load_dotenv
from azure.cosmos import CosmosClient, exceptions
import json
import tiktoken
from openai import AzureOpenAI
from datetime import datetime, timedelta, timezone
import mysql.connector

# .env ファイルをロードして環境変数を取得
# dotenv_path = join(dirname(__file__), '.env')
load_dotenv()

# 環境変数を読み込む
COSMOS_DB_ENDPOINT = os.environ.get("COSMOS_DB_ENDPOINT")
COSMOS_DB_KEY = os.environ.get("COSMOS_DB_KEY")
COSMOS_DB_ID = os.environ.get("COSMOS_DB_ID")
COSMOS_CON_ID = "production-tech0-manager"
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
GPT_MODEL = os.environ.get("GPT_MODEL")
DB_USER = os.environ.get("DB_USER")
DB_PASS = os.environ.get("DB_PASS")
DB_HOST = os.environ.get("DB_HOST")
DB_NAME = os.environ.get("DB_NAME")


app = func.FunctionApp()

@app.timer_trigger(schedule="0 */1 * * * *", arg_name="myTimer", run_on_startup=False,
              use_monitor=False) 
def timer_func01(myTimer: func.TimerRequest) -> None:
    
    if myTimer.past_due:
        logging.info('The timer is past due!')

        # Cosmos DB クライアントを初期化
    client = CosmosClient(COSMOS_DB_ENDPOINT, COSMOS_DB_KEY)
    database = client.get_database_client(COSMOS_DB_ID)
    container = database.get_container_client(COSMOS_CON_ID)

    # Azure OpenAI APIクライアントの初期化
    azure_client = AzureOpenAI(
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        api_key=AZURE_OPENAI_API_KEY,
        api_version="2024-05-01-preview",
    )

    # Azure Database for MySQL connection details
    config = {
        'user': DB_USER,
        'password': DB_PASS,
        'host': DB_HOST,
        'database': DB_NAME,
    }

    def read_data_onehalf():
        current_time = datetime.now(timezone.utc)
        time_threshold = current_time - timedelta(hours=1.5)
        time_threshold_unix = int(time_threshold.timestamp())

        query = f"SELECT * FROM c WHERE c._ts >= {time_threshold_unix}"
        items = list(container.query_items(
            query=query,
            enable_cross_partition_query=True
        ))
        return items

    def create_label(text, labels: dict):
        prompt = f"""
        あなたは優秀なデータアナリストです。プログラミングスクールの受講生の質問を解析したいです。
        以下の文章について、適切なカテゴリを設定してください。

        ###記事内容
        {text}

        ###出力フォーマット
        {{"ラベル名":"その説明"}}

        ###条件
        ・既存のラベルと重なった場合は無視してください。
        ・上記記載のjson形式で返すようにしてください。
        ・説明は30文字以内で書いてください。
        ・プログラミング以外の質問については、「その他：プログラミング以外の内容」で返すようにしてください。

        ###既存のラベル
        {labels}
        """

        response = azure_client.chat.completions.create(
            model=GPT_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            response_format={"type": "json_object"}
        )

        new_labels = json.loads(response.choices[0].message.content)    
        labels.update(new_labels)

        return labels

    def assign_labels(comment, labels):
        label_prompt = f"""
        コメントに最も適したラベルを提案してください。
        次に記載するラベルから選択してください。

        ###コメント
        {comment}

        ###条件
        ・出力形式で指定したjson型でリストにして返すこと。
        ・必要あれば、できる限り多くのラベルを使用してください。
        ・カンマ（,）区切りでjsonで出力してください。例：python, nextjs
        ・適切なラベルがなければ、空のリストを返してください。絶対に、出力軽視の例をそのまま返さないように。
        
        ###ラベル候補
        {labels}

        ###出力形式
        {{labels：["ラベル名1", "ラベル名2","ラベル名3"]}}
        """
        try:
            response = azure_client.chat.completions.create(
                model=GPT_MODEL,
                messages=[
                    {
                        "role": "user",
                        "content": label_prompt
                    }
                ],
                response_format={"type": "json_object"}
            )
            response_content = response.choices[0].message.content.strip()
            assigned_labels = json.loads(response_content)
            return assigned_labels["labels"]
        except Exception as e:
            logging.error(f"API呼び出しでエラーが発生しました: {e}")
            return []

    def conversation_categorize(labels: dict):
        try:
            items = read_data_onehalf()
            for item in items:
                if "category" not in item and 'messages' in item:
                    user_messages = [msg['content'] for msg in item['messages'] if msg['role'] == 'user'][:3]
                    if not user_messages:
                        continue

                    answer = json.dumps(user_messages, ensure_ascii=False)
                    encoding = tiktoken.encoding_for_model("gpt-4-turbo")
                    tokens = encoding.encode(answer)
                    tokens_count = len(tokens)

                    token_limit = 10000

                    texts = []
                    current_tokens = []

                    for token in tokens:
                        current_tokens.append(token)
                        if len(current_tokens) >= token_limit:
                            current_text = encoding.decode(current_tokens)
                            texts.append(current_text)
                            current_tokens = []

                    if current_tokens:
                        current_text = encoding.decode(current_tokens)
                        texts.append(current_text)

                    for text in texts:
                        labels = create_label(text, labels)

                    all_assigned_labels = []

                    for text in texts:
                        assigned_labels = assign_labels(text, labels)
                        if isinstance(assigned_labels, list):
                            all_assigned_labels.extend(assigned_labels)

                    item['category'] = all_assigned_labels if isinstance(all_assigned_labels, list) else []
                    container.upsert_item(item)

        except exceptions.CosmosHttpResponseError as e:
            logging.error(f'Error reading data from Cosmos DB: {e}')

        return labels

    def save_labels_to_db(labels):
        try:
            conn = mysql.connector.connect(**config)
            cursor = conn.cursor()

            create_table_query = '''
            CREATE TABLE IF NOT EXISTS labels (
                id INT AUTO_INCREMENT PRIMARY KEY,
                label_key VARCHAR(255) NOT NULL,
                label_value VARCHAR(255) NOT NULL,
                UNIQUE KEY unique_label (label_key)
            )
            '''
            cursor.execute(create_table_query)

            for key, value in labels.items():
                select_query = 'SELECT COUNT(*) FROM labels WHERE label_key = %s'
                cursor.execute(select_query, (key,))
                result = cursor.fetchone()

                if result[0] == 0:
                    insert_query = '''
                    INSERT INTO labels (label_key, label_value)
                    VALUES (%s, %s)
                    '''
                    cursor.execute(insert_query, (key, value))
                else:
                    logging.info(f"Key '{key}' already exists. Skipping insertion.")

            conn.commit()

        finally:
            cursor.close()
            conn.close()

    #ここから実行
    labels = {}

    try:
        conn = mysql.connector.connect(**config)
        cursor = conn.cursor()

        select_query = 'SELECT label_key, label_value FROM labels'
        cursor.execute(select_query)
        rows = cursor.fetchall()

        for row in rows:
            labels[row[0]] = row[1]

    finally:
        cursor.close()
        conn.close()

    labels = conversation_categorize(labels)
    save_labels_to_db(labels)

    logging.info('Python timer trigger function executed.')