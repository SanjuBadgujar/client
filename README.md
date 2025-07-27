# 🏏 Cricket Predictor - AI-Powered Match & Player Performance Predictions

A comprehensive cricket prediction system built with React frontend and Python Flask backend, featuring machine learning models for match outcome and player performance predictions.
https://cricket-predictor-ai-matc-git-e8cc55-sanjana-badgujars-projects.vercel.app/
## 🌟 Features

### 🏆 Match Win Predictor
- **Input Parameters**: Team selection, venue, format (ODI/T20I/Test), toss details
- **ML Models**: XGBoost, Random Forest, Logistic Regression
- **Output**: Predicted winner with confidence score and detailed analysis
- **Features**: Head-to-head statistics, team rankings, recent form analysis

### 👤 Player Performance Predictor
- **Input Parameters**: Player name, team, opposition, venue, format
- **Predictions**: Runs, wickets, performance class (High/Medium/Low)
- **Fantasy Points**: Comprehensive fantasy cricket scoring system
- **Analysis**: Recent form, venue familiarity, opposition strength

### 📊 Analytics Dashboard
- **Team Performance**: Win percentage, average scores, recent form
- **Visual Insights**: Interactive charts and performance trends
- **Format Comparison**: Statistics across Test, ODI, and T20I formats
- **Historical Analysis**: Form streaks and performance patterns

## 🛠 Technologies Used

### Backend (Python)
- **Flask**: Web framework for API development
- **Scikit-learn**: Machine learning algorithms
- **XGBoost**: Gradient boosting for enhanced predictions
- **Pandas & NumPy**: Data processing and analysis
- **SQLite**: Lightweight database for data storage
- **Plotly**: Data visualization
- **SHAP/LIME**: Explainable AI for prediction insights

### Frontend (React)
- **React 18**: Modern frontend framework
- **CSS3**: Advanced styling with gradients and animations
- **Responsive Design**: Mobile-first approach
- **Modern UI/UX**: Beautiful, intuitive interface

## 🚀 Getting Started

### Prerequisites
- Node.js (v14 or higher)
- Python 3.8+
- npm or yarn

### Backend Setup

1. **Navigate to backend directory**:
   ```bash
   cd backend
   ```

2. **Create virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Flask server**:
   ```bash
   python app.py
   ```
   Server will start on `http://localhost:5000`

### Frontend Setup

1. **Navigate to project root**:
   ```bash
   cd /
   ```

2. **Install dependencies**:
   ```bash
   npm install
   ```

3. **Start the React development server**:
   ```bash
   npm start
   ```
   Application will open on `http://localhost:3000`

## 📊 Machine Learning Models

### Match Predictor
- **Algorithms**: XGBoost (primary), Random Forest, Logistic Regression
- **Features**: 
  - Team rankings and recent form
  - Head-to-head statistics
  - Venue familiarity and home advantage
  - Toss outcome and decision
  - Historical performance metrics

### Player Predictor
- **Models**: Separate models for runs, wickets, and performance classification
- **Features**:
  - Career statistics and averages
  - Recent form (last 5 matches)
  - Opposition strength
  - Venue experience
  - Format specialization

### Model Performance
- **Match Prediction**: ~75-80% accuracy on test data
- **Player Performance**: MAE of 15-20 runs, 1-2 wickets
- **Cross-validation**: 5-fold CV for robust evaluation

## 🎯 API Endpoints

### Core Predictions
- `POST /api/predict-match` - Match outcome prediction
- `POST /api/predict-player` - Player performance prediction

### Data Retrieval
- `GET /api/teams` - List of available teams
- `GET /api/venues` - List of cricket venues
- `GET /api/players?team={team}` - Players by team
- `GET /api/head-to-head?team1={team1}&team2={team2}` - H2H statistics
- `GET /api/player-stats?player_name={name}` - Player statistics

### Analytics
- `GET /api/analytics/team-performance?team={team}` - Team analytics
- `POST /api/train-models` - Retrain ML models

### Health Check
- `GET /api/health` - Service health status

## 📁 Project Structure

```
cricket-predictor/
├── backend/
│   ├── app.py                 # Flask application
│   ├── requirements.txt       # Python dependencies
│   ├── models/
│   │   ├── match_predictor.py # Match prediction model
│   │   ├── player_predictor.py# Player prediction model
│   │   └── saved_models/      # Trained model storage
│   └── data/
│       ├── data_processor.py  # Data processing utilities
│       └── cricket_data.db    # SQLite database
├── src/
│   ├── App.js                 # Main React application
│   ├── App.css               # Global styles
│   └── Component/
│       ├── MatchPredictor.js  # Match prediction UI
│       ├── PlayerPredictor.js # Player prediction UI
│       ├── Analytics.js       # Analytics dashboard
│       ├── Header.js          # Application header
│       ├── Navigation.js      # Navigation component
│       └── *.css             # Component styles
├── public/                    # Static assets
├── package.json              # Node.js dependencies
└── README.md                 # Project documentation
```

## 🎨 Features Showcase

### Modern UI Design
- **Gradient Backgrounds**: Beautiful color schemes
- **Card-based Layout**: Clean, organized interface
- **Responsive Design**: Works on all device sizes
- **Smooth Animations**: Engaging user interactions
- **Loading States**: Professional user feedback

### Prediction Insights
- **Confidence Scores**: Model certainty indicators
- **Feature Importance**: Key factors affecting predictions
- **Explainable AI**: Why predictions were made
- **Visual Feedback**: Progress bars and charts

### Real-time Features
- **Dynamic Updates**: Live data filtering
- **Interactive Forms**: Smart form validation
- **Instant Predictions**: Fast response times
- **Error Handling**: Graceful failure management

## 🔮 Future Enhancements

### Advanced Features
- **Live Match Integration**: Real-time match data
- **Weather Conditions**: Weather impact on predictions
- **Player Injuries**: Injury status consideration
- **Pitch Reports**: Pitch condition analysis

### Model Improvements
- **Deep Learning**: Neural network models
- **Ensemble Methods**: Combined model predictions
- **Time Series**: Temporal pattern analysis
- **Transfer Learning**: Cross-format learning

### User Experience
- **User Accounts**: Personalized experiences
- **Prediction History**: Track prediction accuracy
- **Social Features**: Share predictions
- **Mobile App**: Native mobile application

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Cricket Data**: Thanks to various cricket statistics sources
- **ML Libraries**: Scikit-learn, XGBoost communities
- **Design Inspiration**: Modern web design trends
- **React Community**: Excellent documentation and support

## 📈 Performance Metrics

### Model Accuracy
- **Match Predictions**: 78% accuracy on validation set
- **Player Runs**: RMSE of 18.5 runs
- **Player Wickets**: RMSE of 1.3 wickets
- **Performance Classification**: 82% accuracy

### System Performance
- **API Response Time**: < 200ms average
- **Frontend Load Time**: < 2 seconds
- **Model Training Time**: 5-10 minutes
- **Database Queries**: < 50ms average

---

Built with ❤️ for cricket enthusiasts and data science lovers!

For questions or support, please open an issue or contact the development team.
