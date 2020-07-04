import React from 'react';
import sentimentIm from '../assets/img/portfolio/thumbnails/sentiment_analysis_2.jpg';
import namedEntityIm from '../assets/img/portfolio/thumbnails/NamedEntityRecognition2.jpg';
import clothingClassIm from '../assets/img/portfolio/thumbnails/clothing_classifier.jpeg';

export default class Portfolio extends React.Component {
  render() {
    return (
        <section id="portfolio">
        <div class="container-fluid p-0">
            <div class="row no-gutters">
                <div class="col-lg-4 col-sm-6">
                    <a class="portfolio-box" href="sentimentAnalysis"
                        ><img class="img-fluid" src={sentimentIm} alt="Sentiment Analysis" />
                        <div class="portfolio-box-caption">
                            <div class="project-category text-white-50">Sentiment Analysis</div>
                            <div class="project-name">Measure customer opinions and attitudes through the lens of sentiment analysis</div>
                        </div>
                    </a>
                </div>
                <div class="col-lg-4 col-sm-6">
                    <a class="portfolio-box" href="namedEntityRecognition"
                        ><img class="img-fluid" src={namedEntityIm} alt="Named Entity Recognition" />
                        <div class="portfolio-box-caption">
                            <div class="project-category text-white-50">Named Entity Recognition</div>
                            <div class="project-name">Description for Named Entity Recognition</div> 
                        </div>
                    </a>
                </div>
                <div class="col-lg-4 col-sm-6">
                    <a class="portfolio-box" href="clothingClassifier"
                        ><img class="img-fluid" src={clothingClassIm} alt="Clothing tagging" />
                        <div class="portfolio-box-caption">
                            <div class="project-category text-white-50">Tagging of clothing</div>
                            <div class="project-name">Description for Named Entity Recognition</div> 
                        </div>
                    </a>
                </div>
            </div>
        </div>
    </section>
    );
  }
}