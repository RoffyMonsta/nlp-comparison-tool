import json
import csv
import random
from custom_sentiment import calculate_sentiment

# === Генерація нейтральних даних ===
def generate_neutral_data(count=500):
    templates = [
        "Температура повітря {temp} градусів",
        "Магазин працює з {time1} до {time2}", 
        "Курс долара {rate} гривень",
        "Документи до {date}",
        "Автобус кожні {min} хвилин",
        "У місті {pop} людей",
        "Лекція триватиме {dur} години"
    ]
    
    fillers = {
        'temp': ['20', '15', '25', '18'], 'time1': ['9:00', '10:00'], 'time2': ['18:00', '19:00'],
        'rate': ['37', '38', '39'], 'date': ['понеділка', 'п\'ятниці'], 'min': ['15', '20'],
        'pop': ['50 тисяч', '100 тисяч'], 'dur': ['дві', 'три']
    }
    
    neutral_texts = []
    for _ in range(count):
        template = random.choice(templates)
        text = template
        for key, values in fillers.items():
            if f'{{{key}}}' in text:
                text = text.replace(f'{{{key}}}', random.choice(values))
        
        # Перевіряємо нейтральність
        result = calculate_sentiment(text)
        if abs(result['compound']) < 0.3:
            neutral_texts.append({'msg': text})
    
    return neutral_texts

# === Вхідні файли ===
FILES = [
    ('dict/positive.json', 'positive'),
    ('dict/negative.json', 'negative')
]

# === Вихідні файли ===
OUTPUT_CSV = 'dict/features_dataset.csv'
OUTPUT_NEUTRAL_JSON = 'dict/neutral.json'

# === Заголовки колонок ===
HEADERS = [
    'positive', 'negative', 'neutral', 'compound',
    'emoji_score', 'num_boosters', 'has_negation',
    'num_tokens', 'label'
]

# Генеруємо нейтральні дані
neutral_data = generate_neutral_data(500)
print(f"📊 Згенеровано {len(neutral_data)} нейтральних текстів")

# Зберігаємо нейтральні дані у JSON (як positive.json та negative.json)
with open(OUTPUT_NEUTRAL_JSON, 'w', encoding='utf-8') as f:
    for item in neutral_data:
        f.write(json.dumps(item, ensure_ascii=False) + '\n')
print(f"💾 Нейтральні тексти збережено: {OUTPUT_NEUTRAL_JSON}")

total_lines = len(neutral_data)
for file_path, _ in FILES:
    with open(file_path, encoding='utf-8') as f:
        total_lines += sum(1 for _ in f)

print(f"🔢 Всього твітів до обробки: {total_lines}")

# === Записуємо CSV ===
with open(OUTPUT_CSV, mode='w', newline='', encoding='utf-8') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=HEADERS)
    writer.writeheader()

    processed = 0
    print(f"🔄 Обробка {total_lines} твітів...")
    
    # Обробляємо нейтральні дані
    for obj in neutral_data:
        text = obj.get('msg', '').strip()
        result = calculate_sentiment(text)
        
        row = {
            'positive': result.get('positive', 0),
            'negative': result.get('negative', 0),
            'neutral': result.get('neutral', 0),
            'compound': result.get('compound', 0),
            'emoji_score': result.get('emoji_score', 0),
            'num_boosters': result.get('num_boosters', 0),
            'has_negation': result.get('has_negation', 0),
            'num_tokens': result.get('num_tokens', 0),
            'label': 'neutral'
        }
        writer.writerow(row)
        processed += 1
        if processed % 100 == 0:
            print(f"✅ Оброблено {processed}/{total_lines}")
    
    # Обробляємо існуючі файли
    for file_path, label in FILES:
        with open(file_path, encoding='utf-8') as f:
            for idx, line in enumerate(f, start=1):
                try:
                    obj = json.loads(line)
                    text = obj.get('msg', '').strip()
                    if not text:  # Skip empty texts
                        continue
                except json.JSONDecodeError:
                    print(f"⚠️ Skipping invalid JSON at line {idx} in {file_path}")
                    continue

                result = calculate_sentiment(text)

                row = {
                    'positive': result.get('positive', 0),
                    'negative': result.get('negative', 0),
                    'neutral': result.get('neutral', 0),
                    'compound': result.get('compound', 0),
                    'emoji_score': result.get('emoji_score', 0),
                    'num_boosters': result.get('num_boosters', 0),
                    'has_negation': result.get('has_negation', 0),
                    'num_tokens': result.get('num_tokens', 0),
                    'label': label
                }

                writer.writerow(row)
                processed += 1
                
                if processed % 100 == 0:
                    print(f"✅ Оброблено {processed}/{total_lines}")

print(f"🎉 Готово! CSV збережено: {OUTPUT_CSV}")
