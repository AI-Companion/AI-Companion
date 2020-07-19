import React from 'react';
import sentimentIm from '../assets/img/portfolio/thumbnails/sentiment_analysis_2.jpg';
import namedEntityIm from '../assets/img/portfolio/thumbnails/named_entity_recognition.png';
import fashion_classifier from '../assets/img/portfolio/thumbnails/fashion_classifier.png';

export default class Portfolio extends React.Component {
  render() {
    return (
        <section id="portfolio">
        <div className="container-fluid p-0">
            <div className="row no-gutters">
                <div className="col-lg-4 col-sm-6">
                    <a className="portfolio-box" href="sentimentAnalysis"
                        ><img className="img-fluid" src={sentimentIm} alt="Sentiment Analysis" />
                        <div className="portfolio-box-caption">
                            <div className="project-category text-white-50">Sentiment Analysis</div>
                            <div className="project-name">Measure customer opinions and attitudes through the lens of sentiment analysis</div>
                        </div>
                    </a>
                </div>
                <div className="col-lg-4 col-sm-6">
                    <a className="portfolio-box" href="namedEntityRecognition"
                        ><img className="img-fluid" src={namedEntityIm} alt="Named Entity Recognition" />
                        <div className="portfolio-box-caption">
                            <div className="project-category text-white-50">Named Entity Recognition</div>
                            <div className="project-name">Extract information to identify and segment the named entities or categorize them under various predefined classNamees</div> 
                        </div>
                    </a>
                </div>
                <div className="col-lg-4 col-sm-6">
                    <a className="portfolio-box" href="clothingClassifier"
                        ><img className="img-fluid" src={fashion_classifier} alt="Clothing tagging" />
                        <div className="portfolio-box-caption">
                            <div className="project-category text-white-50">Tagging of clothing</div>
                            <div className="project-name">classNameifying clothing images into different categories using state of the art computer vision algorithms</div> 
                        </div>
                    </a>
                </div>
            </div>
        </div>
    </section>
    );
  }
}