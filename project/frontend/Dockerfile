# =============================================================================
# Многостадийный Docker-образ для React frontend (Vite + Nginx)
# =============================================================================
# Стадия 1 (build): сборка React-приложения
# Стадия 2 (runtime): Nginx для раздачи статики + прокси на API
# =============================================================================

# --- Стадия 1: Сборка React-приложения ---------------------------------------
FROM node:20-alpine AS build

WORKDIR /app

# Копируем манифесты зависимостей и устанавливаем пакеты
COPY package.json ./
RUN npm install && npm cache clean --force

# Копируем исходный код и собираем production-сборку
COPY . ./
RUN npm run build

# --- Стадия 2: Nginx для раздачи статики ------------------------------------
FROM nginx:alpine AS runtime

# Копируем кастомную конфигурацию Nginx (прокси на API)
COPY nginx/default.conf /etc/nginx/conf.d/default.conf

# Копируем собранные статические файлы из стадии build
COPY --from=build /app/dist /usr/share/nginx/html

EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]
