if ! command -v clang-format &> /dev/null; then
    echo "clang-format не установлен. Установите его с помощью: sudo apt install clang-format"
    exit 1
fi

CURRENT_DIR=$(pwd)

find "$CURRENT_DIR" -type f \( -name "*.cpp" -o -name "*.hpp" \) -exec clang-format -i {} \;

echo "Форматирование завершено."