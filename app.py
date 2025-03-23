from flask import Flask, render_template, request
import pandas as pd
import matplotlib.pyplot as plt
import io
import base64
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# Load dataset 
data = pd.read_csv('actual_abroad_data.csv')

# Handle missing values
data.fillna(data.mean(numeric_only=True), inplace=True)  # Replace NaN in numeric columns with the mean
data.fillna("Unknown", inplace=True)  # Replace NaN in categorical columns with "Unknown"

# Helper function to create and encode plots
def create_plot(data, plot_type, feature_x=None, feature_y=None):
    plt.figure(figsize=(6, 4))
    
    if plot_type == 'line':
        data['FEES'].plot(kind='line', title='Line Plot of Fees')
        plt.ylabel('Fees')
        plt.xlabel('Index')
    
    elif plot_type == 'bar':
        data.groupby('COUNTRY')['FEES'].mean().plot(kind='bar', title='Bar Plot of Average Fees by Country')
        plt.ylabel('Average Fees')
        plt.xlabel('Country')
    
    elif plot_type == 'scatter' and feature_x and feature_y:
        plt.scatter(data[feature_x], data[feature_y], c='blue', alpha=0.6)
        plt.title(f'Scatter Plot: {feature_x} vs {feature_y}')
        plt.xlabel(feature_x)
        plt.ylabel(feature_y)
    
    elif plot_type == 'pca':
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(data[['FEES']])
        pca = PCA(n_components=1)
        pca_data = pca.fit_transform(scaled_data)
        plt.plot(pca_data, 'o-', label='PCA Component 1')
        plt.title('PCA Visualization')
        plt.ylabel('PCA Value')
        plt.xlabel('Index')
        plt.legend()
    
    # Save plot to buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode('utf-8')
    buf.close()
    return encoded

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get form data
    country = request.form['country'].strip().lower()
    course_type = request.form['course_type'].strip().lower()
    course_spec = request.form['course_spec'].strip().lower()
    
    # Filter dataset based on input
    filtered_data = data[
        (data['COUNTRY'].str.lower() == country) &
        (data['COURSE TYPE'].str.lower() == course_type) &
        (data['COURSE (SPECIALIZATION)'].str.lower() == course_spec)
    ]
    
    if not filtered_data.empty:
        fees = filtered_data.iloc[0]['FEES']
        cost_category = "High Cost" if fees > 400000 else "Low Cost"
        
        # Generate visualizations
        line_plot = create_plot(data, 'line')
        bar_plot = create_plot(data, 'bar')
        scatter_plot = create_plot(data, 'scatter', 'FEES', 'FEES')
        pca_plot = create_plot(data, 'pca')
        
        return render_template('result.html', fees=fees, category=cost_category, 
                               line_plot=line_plot, bar_plot=bar_plot, 
                               scatter_plot=scatter_plot, pca_plot=pca_plot)
    else:
        return render_template('result.html', error="No matching course found in the dataset.")

if __name__ == '__main__':
    app.run(debug=True)
