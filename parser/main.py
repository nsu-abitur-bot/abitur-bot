import logging

from NSU_parser import parse_nsu

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    print("Запуск парсера НГУ...")
    data = parse_nsu("information-technologies")
    logging.info(data)
    print("Парсинг завершен")

if __name__ == "__main__":
    main()