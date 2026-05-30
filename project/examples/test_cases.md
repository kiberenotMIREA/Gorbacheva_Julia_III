# Примеры для тестирования Vulnerability Scoring API

Модель обучена на датасете VulDeePecker (CWE-119 buffer errors + CWE-399 resource management). Функции C/C++ из Juliet Test Suite / SARD.

Формат ответа: `(prediction, [prob_safe, prob_vuln])`

---

## Тестирование через Python

```python
from src.predict import VulnerabilityPredictor
p = VulnerabilityPredictor()

code = """ваш C/C++ код"""
print(p.predict(code, method="ensemble"))
```

> **Адрес:** `localhost` — для локального запуска, замените на IP-адрес сервера при удаленном подключении.

## Тестирование через API — общие шаблоны

### Способ A: curl + экранированная строка

```bash
curl -s -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"code": "void f() { char buf[10]; strcpy(buf, argv[1]); }", "method": "ensemble"}'
```

### Способ B: curl + временный файл + Python

```bash
cat > /tmp/sample.c << 'EOF'
void f() {
    char buf[10];
    strcpy(buf, argv[1]);
}
EOF
python3 -c "
import json, urllib.request
with open('/tmp/sample.c') as f:
    body = json.dumps({'code': f.read(), 'method': 'ensemble'}).encode()
req = urllib.request.Request('http://localhost:8000/predict', body, {'Content-Type': 'application/json'})
print(urllib.request.urlopen(req).read().decode())
"
```

### Способ C: curl + временный файл + jq

```bash
cat > /tmp/sample.c << 'EOF'
void f() { char buf[10]; strcpy(buf, argv[1]); }
EOF
jq -Rs '{code: ., method: "ensemble"}' /tmp/sample.c | \
  curl -s -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d @-
```

---

## Уязвимые образцы (CWE-119 / CWE-399, должны → VULN)

### 1. Buffer Underread (CWE-119)

```c
void copy_buffer()
{
    char * data;
    char dataBuffer[100];
    memset(dataBuffer, 'A', 100-1);
    dataBuffer[100-1] = '\0';
    data = dataBuffer - 8;
    {
        char dest[100];
        memset(dest, 'C', 100-1);
        dest[100-1] = '\0';
        strncpy(dest, data, strlen(dest));
        dest[100-1] = '\0';
        printLine(dest);
    }
}
```

**A) curl + экранированная строка:**
```bash
curl -s -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"code": "void copy_buffer()\n{\n    char * data;\n    char dataBuffer[100];\n    memset(dataBuffer, '\''A'\'', 100-1);\n    dataBuffer[100-1] = '\''\\0'\'';\n    data = dataBuffer - 8;\n    {\n        char dest[100];\n        memset(dest, '\''C'\'', 100-1);\n        dest[100-1] = '\''\\0'\'';\n        strncpy(dest, data, strlen(dest));\n        dest[100-1] = '\''\\0'\'';\n        printLine(dest);\n    }\n}", "method": "ensemble"}'
```

**B) curl + временный файл + Python:**
```bash
cat > /tmp/vuln_underread.c << 'EOF'
void copy_buffer()
{
    char * data;
    char dataBuffer[100];
    memset(dataBuffer, 'A', 100-1);
    dataBuffer[100-1] = '\0';
    data = dataBuffer - 8;
    {
        char dest[100];
        memset(dest, 'C', 100-1);
        dest[100-1] = '\0';
        strncpy(dest, data, strlen(dest));
        dest[100-1] = '\0';
        printLine(dest);
    }
}
EOF
python3 -c "
import json, urllib.request
with open('/tmp/vuln_underread.c') as f:
    body = json.dumps({'code': f.read(), 'method': 'ensemble'}).encode()
req = urllib.request.Request('http://localhost:8000/predict', body, {'Content-Type': 'application/json'})
print(urllib.request.urlopen(req).read().decode())
"
```

**C) curl + временный файл + jq:**
```bash
cat > /tmp/vuln_underread.c << 'EOF'
void copy_buffer()
{
    char * data;
    char dataBuffer[100];
    memset(dataBuffer, 'A', 100-1);
    dataBuffer[100-1] = '\0';
    data = dataBuffer - 8;
    {
        char dest[100];
        memset(dest, 'C', 100-1);
        dest[100-1] = '\0';
        strncpy(dest, data, strlen(dest));
        dest[100-1] = '\0';
        printLine(dest);
    }
}
EOF
jq -Rs '{code: ., method: "ensemble"}' /tmp/vuln_underread.c | \
  curl -s -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d @-
```

**D) Swagger UI (`/docs`):** вставьте эту строку в поле `code`:

```json
"void copy_buffer()\n{\n    char * data;\n    char dataBuffer[100];\n    memset(dataBuffer, 'A', 100-1);\n    dataBuffer[100-1] = '\\0';\n    data = dataBuffer - 8;\n    {\n        char dest[100];\n        memset(dest, 'C', 100-1);\n        dest[100-1] = '\\0';\n        strncpy(dest, data, strlen(dest));\n        dest[100-1] = '\\0';\n        printLine(dest);\n    }\n}"
```

`"method": "ensemble"`

### 2. Stack Buffer Overflow via strcpy (CWE-119)

```c
void process_data()
{
    char source[100];
    memset(source, 'C', 100-1);
    source[100-1] = '\0';
    {
        char dest[50];
        strcpy(dest, source);
        printLine(dest);
    }
}
```

**A) curl + экранированная строка:**
```bash
curl -s -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"code": "void process_data()\n{\n    char source[100];\n    memset(source, '\''C'\'', 100-1);\n    source[100-1] = '\''\\0'\'';\n    {\n        char dest[50];\n        strcpy(dest, source);\n        printLine(dest);\n    }\n}", "method": "ensemble"}'
```

**B) curl + временный файл + Python:**
```bash
cat > /tmp/vuln_strcpy.c << 'EOF'
void process_data()
{
    char source[100];
    memset(source, 'C', 100-1);
    source[100-1] = '\0';
    {
        char dest[50];
        strcpy(dest, source);
        printLine(dest);
    }
}
EOF
python3 -c "
import json, urllib.request
with open('/tmp/vuln_strcpy.c') as f:
    body = json.dumps({'code': f.read(), 'method': 'ensemble'}).encode()
req = urllib.request.Request('http://localhost:8000/predict', body, {'Content-Type': 'application/json'})
print(urllib.request.urlopen(req).read().decode())
"
```

**C) curl + временный файл + jq:**
```bash
cat > /tmp/vuln_strcpy.c << 'EOF'
void process_data()
{
    char source[100];
    memset(source, 'C', 100-1);
    source[100-1] = '\0';
    {
        char dest[50];
        strcpy(dest, source);
        printLine(dest);
    }
}
EOF
jq -Rs '{code: ., method: "ensemble"}' /tmp/vuln_strcpy.c | \
  curl -s -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d @-
```

**D) Swagger UI (`/docs`):** вставьте эту строку в поле `code`:

```json
"void process_data()\n{\n    char source[100];\n    memset(source, 'C', 100-1);\n    source[100-1] = '\\0';\n    {\n        char dest[50];\n        strcpy(dest, source);\n        printLine(dest);\n    }\n}"
```

`"method": "ensemble"`

### 3. Stack Buffer Overflow via wcscpy (CWE-119)

```c
void merge_strings()
{
    wchar_t * data;
    wchar_t * dataBuffer = (wchar_t *)ALLOCA(100*sizeof(wchar_t));
    data = dataBuffer;
    wmemset(data, L'A', 100-1);
    data[100-1] = L'\0';
    {
        wchar_t dest[50] = L"";
        wcscpy(dest, data);
        printWLine(data);
    }
}
```

**A) curl + экранированная строка:**
```bash
curl -s -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"code": "void merge_strings()\n{\n    wchar_t * data;\n    wchar_t * dataBuffer = (wchar_t *)ALLOCA(100*sizeof(wchar_t));\n    data = dataBuffer;\n    wmemset(data, L'\''A'\'', 100-1);\n    data[100-1] = L'\''\\0'\'';\n    {\n        wchar_t dest[50] = L\"\";\n        wcscpy(dest, data);\n        printWLine(data);\n    }\n}", "method": "ensemble"}'
```

**B) curl + временный файл + Python:**
```bash
cat > /tmp/vuln_wcscpy.c << 'EOF'
void merge_strings()
{
    wchar_t * data;
    wchar_t * dataBuffer = (wchar_t *)ALLOCA(100*sizeof(wchar_t));
    data = dataBuffer;
    wmemset(data, L'A', 100-1);
    data[100-1] = L'\0';
    {
        wchar_t dest[50] = L"";
        wcscpy(dest, data);
        printWLine(data);
    }
}
EOF
python3 -c "
import json, urllib.request
with open('/tmp/vuln_wcscpy.c') as f:
    body = json.dumps({'code': f.read(), 'method': 'ensemble'}).encode()
req = urllib.request.Request('http://localhost:8000/predict', body, {'Content-Type': 'application/json'})
print(urllib.request.urlopen(req).read().decode())
"
```

**C) curl + временный файл + jq:**
```bash
cat > /tmp/vuln_wcscpy.c << 'EOF'
void merge_strings()
{
    wchar_t * data;
    wchar_t * dataBuffer = (wchar_t *)ALLOCA(100*sizeof(wchar_t));
    data = dataBuffer;
    wmemset(data, L'A', 100-1);
    data[100-1] = L'\0';
    {
        wchar_t dest[50] = L"";
        wcscpy(dest, data);
        printWLine(data);
    }
}
EOF
jq -Rs '{code: ., method: "ensemble"}' /tmp/vuln_wcscpy.c | \
  curl -s -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d @-
```

**D) Swagger UI (`/docs`):** вставьте эту строку в поле `code`:

```json
"void merge_strings()\n{\n    wchar_t * data;\n    wchar_t * dataBuffer = (wchar_t *)ALLOCA(100*sizeof(wchar_t));\n    data = dataBuffer;\n    wmemset(data, L'A', 100-1);\n    data[100-1] = L'\\0';\n    {\n        wchar_t dest[50] = L\"\";\n        wcscpy(dest, data);\n        printWLine(data);\n    }\n}"
```

`"method": "ensemble"`

### 4. Use After Free (CWE-399)

```c
void handle_data()
{
    int * data;
    data = NULL;
    data = (int *)malloc(100*sizeof(int));
    if (data == NULL) {exit(-1);}
    free(data);
    printIntLine(data[0]);
}
```

**A) curl + экранированная строка:**
```bash
curl -s -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"code": "void handle_data()\n{\n    int * data;\n    data = NULL;\n    data = (int *)malloc(100*sizeof(int));\n    if (data == NULL) {exit(-1);}\n    free(data);\n    printIntLine(data[0]);\n}", "method": "ensemble"}'
```

**B) curl + временный файл + Python:**
```bash
cat > /tmp/vuln_uaf.c << 'EOF'
void handle_data()
{
    int * data;
    data = NULL;
    data = (int *)malloc(100*sizeof(int));
    if (data == NULL) {exit(-1);}
    free(data);
    printIntLine(data[0]);
}
EOF
python3 -c "
import json, urllib.request
with open('/tmp/vuln_uaf.c') as f:
    body = json.dumps({'code': f.read(), 'method': 'ensemble'}).encode()
req = urllib.request.Request('http://localhost:8000/predict', body, {'Content-Type': 'application/json'})
print(urllib.request.urlopen(req).read().decode())
"
```

**C) curl + временный файл + jq:**
```bash
cat > /tmp/vuln_uaf.c << 'EOF'
void handle_data()
{
    int * data;
    data = NULL;
    data = (int *)malloc(100*sizeof(int));
    if (data == NULL) {exit(-1);}
    free(data);
    printIntLine(data[0]);
}
EOF
jq -Rs '{code: ., method: "ensemble"}' /tmp/vuln_uaf.c | \
  curl -s -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d @-
```

**D) Swagger UI (`/docs`):** вставьте эту строку в поле `code`:

```json
"void handle_data()\n{\n    int * data;\n    data = NULL;\n    data = (int *)malloc(100*sizeof(int));\n    if (data == NULL) {exit(-1);}\n    free(data);\n    printIntLine(data[0]);\n}"
```

`"method": "ensemble"`

### 5. Double Free (CWE-399)

```c
void cleanup_resources()
{
    int * data;
    data = NULL;
    data = (int *)malloc(100*sizeof(int));
    if (data == NULL) {exit(-1);}
    free(data);
    free(data);
}
```

**A) curl + экранированная строка:**
```bash
curl -s -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"code": "void cleanup_resources()\n{\n    int * data;\n    data = NULL;\n    data = (int *)malloc(100*sizeof(int));\n    if (data == NULL) {exit(-1);}\n    free(data);\n    free(data);\n}", "method": "ensemble"}'
```

**B) curl + временный файл + Python:**
```bash
cat > /tmp/vuln_doublefree.c << 'EOF'
void cleanup_resources()
{
    int * data;
    data = NULL;
    data = (int *)malloc(100*sizeof(int));
    if (data == NULL) {exit(-1);}
    free(data);
    free(data);
}
EOF
python3 -c "
import json, urllib.request
with open('/tmp/vuln_doublefree.c') as f:
    body = json.dumps({'code': f.read(), 'method': 'ensemble'}).encode()
req = urllib.request.Request('http://localhost:8000/predict', body, {'Content-Type': 'application/json'})
print(urllib.request.urlopen(req).read().decode())
"
```

**C) curl + временный файл + jq:**
```bash
cat > /tmp/vuln_doublefree.c << 'EOF'
void cleanup_resources()
{
    int * data;
    data = NULL;
    data = (int *)malloc(100*sizeof(int));
    if (data == NULL) {exit(-1);}
    free(data);
    free(data);
}
EOF
jq -Rs '{code: ., method: "ensemble"}' /tmp/vuln_doublefree.c | \
  curl -s -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d @-
```

**D) Swagger UI (`/docs`):** вставьте эту строку в поле `code`:

```json
"void cleanup_resources()\n{\n    int * data;\n    data = NULL;\n    data = (int *)malloc(100*sizeof(int));\n    if (data == NULL) {exit(-1);}\n    free(data);\n    free(data);\n}"
```

`"method": "ensemble"`

---

## Безопасные образцы (должны → SAFE)

### 1. strncpy с проверкой границ

```c
void safe_copy()
{
    char * data;
    char dataBuffer[100];
    memset(dataBuffer, 'A', 100-1);
    dataBuffer[100-1] = '\0';
    data = dataBuffer;
    {
        char source[100];
        memset(source, 'C', 100-1);
        source[100-1] = '\0';
        strncpy(data, source, 100-1);
        data[100-1] = '\0';
        printLine(data);
    }
}
```

**A) curl + экранированная строка:**
```bash
curl -s -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"code": "void safe_copy()\n{\n    char * data;\n    char dataBuffer[100];\n    memset(dataBuffer, '\''A'\'', 100-1);\n    dataBuffer[100-1] = '\''\\0'\'';\n    data = dataBuffer;\n    {\n        char source[100];\n        memset(source, '\''C'\'', 100-1);\n        source[100-1] = '\''\\0'\'';\n        strncpy(data, source, 100-1);\n        data[100-1] = '\''\\0'\'';\n        printLine(data);\n    }\n}", "method": "ensemble"}'
```

**B) curl + временный файл + Python:**
```bash
cat > /tmp/safe_strncpy.c << 'EOF'
void safe_copy()
{
    char * data;
    char dataBuffer[100];
    memset(dataBuffer, 'A', 100-1);
    dataBuffer[100-1] = '\0';
    data = dataBuffer;
    {
        char source[100];
        memset(source, 'C', 100-1);
        source[100-1] = '\0';
        strncpy(data, source, 100-1);
        data[100-1] = '\0';
        printLine(data);
    }
}
EOF
python3 -c "
import json, urllib.request
with open('/tmp/safe_strncpy.c') as f:
    body = json.dumps({'code': f.read(), 'method': 'ensemble'}).encode()
req = urllib.request.Request('http://localhost:8000/predict', body, {'Content-Type': 'application/json'})
print(urllib.request.urlopen(req).read().decode())
"
```

**C) curl + временный файл + jq:**
```bash
cat > /tmp/safe_strncpy.c << 'EOF'
void safe_copy()
{
    char * data;
    char dataBuffer[100];
    memset(dataBuffer, 'A', 100-1);
    dataBuffer[100-1] = '\0';
    data = dataBuffer;
    {
        char source[100];
        memset(source, 'C', 100-1);
        source[100-1] = '\0';
        strncpy(data, source, 100-1);
        data[100-1] = '\0';
        printLine(data);
    }
}
EOF
jq -Rs '{code: ., method: "ensemble"}' /tmp/safe_strncpy.c | \
  curl -s -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d @-
```

**D) Swagger UI (`/docs`):** вставьте эту строку в поле `code`:

```json
"void safe_copy()\n{\n    char * data;\n    char dataBuffer[100];\n    memset(dataBuffer, 'A', 100-1);\n    dataBuffer[100-1] = '\\0';\n    data = dataBuffer;\n    {\n        char source[100];\n        memset(source, 'C', 100-1);\n        source[100-1] = '\\0';\n        strncpy(data, source, 100-1);\n        data[100-1] = '\\0';\n        printLine(data);\n    }\n}"
```

`"method": "ensemble"`

### 2. Простая функция без уязвимостей

```c
static PRUnichar*
safe_expand(PRUnichar* aDest, PRUint8* aSrc, PRUint32 aCount)
{
  while (aCount) {
    *aDest = *aSrc;
    ++aDest;
    ++aSrc;
    --aCount;
  }
  return aDest;
}
```

**A) curl + экранированная строка:**
```bash
curl -s -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"code": "static PRUnichar*\nsafe_expand(PRUnichar* aDest, PRUint8* aSrc, PRUint32 aCount)\n{\n  while (aCount) {\n    *aDest = *aSrc;\n    ++aDest;\n    ++aSrc;\n    --aCount;\n  }\n  return aDest;\n}", "method": "ensemble"}'
```

**B) curl + временный файл + Python:**
```bash
cat > /tmp/safe_expand.c << 'EOF'
static PRUnichar*
safe_expand(PRUnichar* aDest, PRUint8* aSrc, PRUint32 aCount)
{
  while (aCount) {
    *aDest = *aSrc;
    ++aDest;
    ++aSrc;
    --aCount;
  }
  return aDest;
}
EOF
python3 -c "
import json, urllib.request
with open('/tmp/safe_expand.c') as f:
    body = json.dumps({'code': f.read(), 'method': 'ensemble'}).encode()
req = urllib.request.Request('http://localhost:8000/predict', body, {'Content-Type': 'application/json'})
print(urllib.request.urlopen(req).read().decode())
"
```

**C) curl + временный файл + jq:**
```bash
cat > /tmp/safe_expand.c << 'EOF'
static PRUnichar*
safe_expand(PRUnichar* aDest, PRUint8* aSrc, PRUint32 aCount)
{
  while (aCount) {
    *aDest = *aSrc;
    ++aDest;
    ++aSrc;
    --aCount;
  }
  return aDest;
}
EOF
jq -Rs '{code: ., method: "ensemble"}' /tmp/safe_expand.c | \
  curl -s -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d @-
```

**D) Swagger UI (`/docs`):** вставьте эту строку в поле `code`:

```json
"static PRUnichar*\nsafe_expand(PRUnichar* aDest, PRUint8* aSrc, PRUint32 aCount)\n{\n  while (aCount) {\n    *aDest = *aSrc;\n    ++aDest;\n    ++aSrc;\n    --aCount;\n  }\n  return aDest;\n}"
```

`"method": "ensemble"`

### 3. Main с безопасной обработкой аргументов

```c
int main(int argc, char* argv[])
{
    if(argv[1])
    {
        int val = atoi(argv[1]);
        printf("val = %d\n", val);
    }
    else
    {
        printf("Usage: %s <number>\n", argv[0]);
    }
    return 0;
}
```

**A) curl + экранированная строка:**
```bash
curl -s -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"code": "int main(int argc, char* argv[])\n{\n    if(argv[1])\n    {\n        int val = atoi(argv[1]);\n        printf(\"val = %d\\n\", val);\n    }\n    else\n    {\n        printf(\"Usage: %s <number>\\n\", argv[0]);\n    }\n    return 0;\n}", "method": "ensemble"}'
```

**B) curl + временный файл + Python:**
```bash
cat > /tmp/safe_main.c << 'EOF'
int main(int argc, char* argv[])
{
    if(argv[1])
    {
        int val = atoi(argv[1]);
        printf("val = %d\n", val);
    }
    else
    {
        printf("Usage: %s <number>\n", argv[0]);
    }
    return 0;
}
EOF
python3 -c "
import json, urllib.request
with open('/tmp/safe_main.c') as f:
    body = json.dumps({'code': f.read(), 'method': 'ensemble'}).encode()
req = urllib.request.Request('http://localhost:8000/predict', body, {'Content-Type': 'application/json'})
print(urllib.request.urlopen(req).read().decode())
"
```

**C) curl + временный файл + jq:**
```bash
cat > /tmp/safe_main.c << 'EOF'
int main(int argc, char* argv[])
{
    if(argv[1])
    {
        int val = atoi(argv[1]);
        printf("val = %d\n", val);
    }
    else
    {
        printf("Usage: %s <number>\n", argv[0]);
    }
    return 0;
}
EOF
jq -Rs '{code: ., method: "ensemble"}' /tmp/safe_main.c | \
  curl -s -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d @-
```

**D) Swagger UI (`/docs`):** вставьте эту строку в поле `code`:

```json
"int main(int argc, char* argv[])\n{\n    if(argv[1])\n    {\n        int val = atoi(argv[1]);\n        printf(\"val = %d\\n\", val);\n    }\n    else\n    {\n        printf(\"Usage: %s <number>\\n\", argv[0]);\n    }\n    return 0;\n}"
```

`"method": "ensemble"`

### 4. Безопасное выделение и освобождение

```c
void safe_alloc_free()
{
    int * data;
    data = NULL;
    data = (int *)malloc(100*sizeof(int));
    if (data == NULL) {exit(-1);}
    free(data);
    data = NULL;
}
```

**A) curl + экранированная строка:**
```bash
curl -s -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"code": "void safe_alloc_free()\n{\n    int * data;\n    data = NULL;\n    data = (int *)malloc(100*sizeof(int));\n    if (data == NULL) {exit(-1);}\n    free(data);\n    data = NULL;\n}", "method": "ensemble"}'
```

**B) curl + временный файл + Python:**
```bash
cat > /tmp/safe_alloc.c << 'EOF'
void safe_alloc_free()
{
    int * data;
    data = NULL;
    data = (int *)malloc(100*sizeof(int));
    if (data == NULL) {exit(-1);}
    free(data);
    data = NULL;
}
EOF
python3 -c "
import json, urllib.request
with open('/tmp/safe_alloc.c') as f:
    body = json.dumps({'code': f.read(), 'method': 'ensemble'}).encode()
req = urllib.request.Request('http://localhost:8000/predict', body, {'Content-Type': 'application/json'})
print(urllib.request.urlopen(req).read().decode())
"
```

**C) curl + временный файл + jq:**
```bash
cat > /tmp/safe_alloc.c << 'EOF'
void safe_alloc_free()
{
    int * data;
    data = NULL;
    data = (int *)malloc(100*sizeof(int));
    if (data == NULL) {exit(-1);}
    free(data);
    data = NULL;
}
EOF
jq -Rs '{code: ., method: "ensemble"}' /tmp/safe_alloc.c | \
  curl -s -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d @-
```

**D) Swagger UI (`/docs`):** вставьте эту строку в поле `code`:

```json
"void safe_alloc_free()\n{\n    int * data;\n    data = NULL;\n    data = (int *)malloc(100*sizeof(int));\n    if (data == NULL) {exit(-1);}\n    free(data);\n    data = NULL;\n}"
```

`"method": "ensemble"`

### 5. Функция без опасных операций

```c
void
nsDocument::notifyStyleRuleChanged(nsIStyleSheet* aStyleSheet,
                                   nsIStyleRule* aOldStyleRule,
                                   nsIStyleRule* aNewStyleRule)
{
  NS_DOCUMENT_NOTIFY_OBSERVERS(StyleRuleChanged,
                                (this, aStyleSheet,
                                 aOldStyleRule, aNewStyleRule));
}
```

**A) curl + экранированная строка:**
```bash
curl -s -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"code": "void\nnsDocument::notifyStyleRuleChanged(nsIStyleSheet* aStyleSheet,\n                                   nsIStyleRule* aOldStyleRule,\n                                   nsIStyleRule* aNewStyleRule)\n{\n  NS_DOCUMENT_NOTIFY_OBSERVERS(StyleRuleChanged,\n                                (this, aStyleSheet,\n                                 aOldStyleRule, aNewStyleRule));\n}", "method": "ensemble"}'
```

**B) curl + временный файл + Python:**
```bash
cat > /tmp/safe_notify.c << 'EOF'
void
nsDocument::notifyStyleRuleChanged(nsIStyleSheet* aStyleSheet,
                                   nsIStyleRule* aOldStyleRule,
                                   nsIStyleRule* aNewStyleRule)
{
  NS_DOCUMENT_NOTIFY_OBSERVERS(StyleRuleChanged,
                                (this, aStyleSheet,
                                 aOldStyleRule, aNewStyleRule));
}
EOF
python3 -c "
import json, urllib.request
with open('/tmp/safe_notify.c') as f:
    body = json.dumps({'code': f.read(), 'method': 'ensemble'}).encode()
req = urllib.request.Request('http://localhost:8000/predict', body, {'Content-Type': 'application/json'})
print(urllib.request.urlopen(req).read().decode())
"
```

**C) curl + временный файл + jq:**
```bash
cat > /tmp/safe_notify.c << 'EOF'
void
nsDocument::notifyStyleRuleChanged(nsIStyleSheet* aStyleSheet,
                                   nsIStyleRule* aOldStyleRule,
                                   nsIStyleRule* aNewStyleRule)
{
  NS_DOCUMENT_NOTIFY_OBSERVERS(StyleRuleChanged,
                                (this, aStyleSheet,
                                 aOldStyleRule, aNewStyleRule));
}
EOF
jq -Rs '{code: ., method: "ensemble"}' /tmp/safe_notify.c | \
  curl -s -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d @-
```

**D) Swagger UI (`/docs`):** вставьте эту строку в поле `code`:

```json
"void\nnsDocument::notifyStyleRuleChanged(nsIStyleSheet* aStyleSheet,\n                                   nsIStyleRule* aOldStyleRule,\n                                   nsIStyleRule* aNewStyleRule)\n{\n  NS_DOCUMENT_NOTIFY_OBSERVERS(StyleRuleChanged,\n                                (this, aStyleSheet,\n                                 aOldStyleRule, aNewStyleRule));\n}"
```

`"method": "ensemble"`

---

## Проверка работоспособности

```bash
# Инференс через Python
python -c "
from src.predict import VulnerabilityPredictor
p = VulnerabilityPredictor()
code = '''void bad() {
    int * data = (int *)malloc(100*sizeof(int));
    free(data);
    free(data);
}'''
print(p.predict(code, method='ensemble'))
"
```

---

## Примечания

- Модель ожидает **полные функции C/C++** (с сигнатурой, телом, скобками)
- Короткие строки (`free(data);`) без контекста не распознаются
- Метод `ensemble` (по умолчанию) — ансамбль Stacking (BERT + LSTM)
- Метод `bert` — DistilBERT + LogisticRegression (F1=0.578)
- Порог обнаружения ensemble: `0.81` (ensemble_meta_thresh.npy)
- Все 10 примеров валидированы на сервере — проходят 10/10
