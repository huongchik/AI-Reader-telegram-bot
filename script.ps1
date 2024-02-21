$botScript = "botAI.py"

while($true) {
    try {
        python $botScript
    }
    catch {
        # Обработка ошибки (если необходимо)
    }
    Start-Sleep -Seconds 5  # Задержка перед следующим запуском
}