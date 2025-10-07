import logging

from nsu_parser import parse_nsu_faculty

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    print("Запуск парсера НГУ...")
    data = parse_nsu_faculty("information-technologies/")
    logging.info(data)
    print("Парсинг завершен")

if __name__ == "__main__":
    main()