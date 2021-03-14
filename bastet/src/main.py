import logging as lg
from src.scrapperManager import ScrapperManager

_logger = lg.getLogger(__name__)


def setup_logging():
    # capture warnings issued by the warnings module
    lg.captureWarnings(True)

    logger = lg.getLogger()
    logger.setLevel(lg.DEBUG)

    # Configure stream logging if applicable
    stream_handler = lg.StreamHandler()
    stream_handler.setLevel(lg.INFO)

    fmt = "%(asctime)s %(levelname)s %(name)s: %(message)s"
    stream_handler.setFormatter(lg.Formatter(fmt))
    logger.addHandler(stream_handler)


def main():
    # config = {
    #     "simpleCurr": ["eur","gbp","cad","jpy","nzd","chf","aud","sar","kyd","czk","isk","rub"]
    # }
    try:
        setup_logging()
        manager = ScrapperManager()
        manager.start()
    except:
        manager.stop()


if __name__ == '__main__':
    main()
