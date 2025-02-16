
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
import seaborn as sns
from scipy import stats

class ModelEvaluationVisualizer:
    """Advanced model evaluation visualization from our experiments."""
    
    def plot_error_distribution(self, true, predicted, bins=50):
        """Plot the distribution of prediction errors."""
        errors = predicted - true
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Histogram
        sns.histplot(errors.flatten(), bins=bins, ax=ax1)
        ax1.set_title("Error Distribution")
        ax1.set_xlabel("Error")
        ax1.set_ylabel("Count")
        
        # Q-Q plot
        stats.probplot(errors.flatten(), dist="norm", plot=ax2)
        ax2.set_title("Q-Q Plot")
        
        plt.tight_layout()
        return fig, (ax1, ax2)
    
    def plot_error_metrics(self, true, predicted, by_latitude=True):
        """Plot error metrics by latitude or longitude."""
        if by_latitude:
            axis = 0
            coord_name = "Latitude"
        else:
            axis = 1
            coord_name = "Longitude"
        
        mse = np.mean((predicted - true)**2, axis=axis)
        mae = np.mean(np.abs(predicted - true), axis=axis)
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        
        # MSE plot
        ax1.plot(mse)
        ax1.set_title(f"MSE by {coord_name}")
        ax1.set_xlabel(coord_name)
        ax1.set_ylabel("MSE")
        
        # MAE plot
        ax2.plot(mae)
        ax2.set_title(f"MAE by {coord_name}")
        ax2.set_xlabel(coord_name)
        ax2.set_ylabel("MAE")
        
        plt.tight_layout()
        return fig, (ax1, ax2)
    
    def plot_temporal_performance(self, true_sequence, pred_sequence,
                                metric='rmse'):
        """Plot model performance over time."""
        if metric == 'rmse':
            scores = [np.sqrt(mean_squared_error(true, pred))
                     for true, pred in zip(true_sequence, pred_sequence)]
        elif metric == 'mae':
            scores = [mean_absolute_error(true, pred)
                     for true, pred in zip(true_sequence, pred_sequence)]
        
        plt.figure(figsize=(12, 5))
        plt.plot(scores)
        plt.title(f"Model Performance Over Time ({metric.upper()})")
        plt.xlabel("Time Step")
        plt.ylabel(metric.upper())
        plt.grid(True)
        
        return plt.gcf(), plt.gca()
    
    def plot_feature_importance(self, importance_scores, feature_names):
        """Plot feature importance scores."""
        plt.figure(figsize=(12, 6))
        
        # Sort by importance
        sorted_idx = np.argsort(importance_scores)
        pos = np.arange(sorted_idx.shape[0]) + .5
        
        plt.barh(pos, importance_scores[sorted_idx])
        plt.yticks(pos, np.array(feature_names)[sorted_idx])
        plt.xlabel('Feature Importance Score')
        plt.title('Feature Importance Analysis')
        
        return plt.gcf(), plt.gca()
    
    def plot_spatial_error_patterns(self, true, predicted,
                                  lat_grid=None, lon_grid=None):
        """Visualize spatial patterns in prediction errors."""
        error = np.abs(predicted - true)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Error magnitude map
        if lat_grid is not None and lon_grid is not None:
            X, Y = np.meshgrid(lon_grid, lat_grid)
            cs1 = ax1.contourf(X, Y, error)
        else:
            cs1 = ax1.imshow(error)
        plt.colorbar(cs1, ax=ax1)
        ax1.set_title("Spatial Error Pattern")
        
        # Error correlation with latitude
        if lat_grid is not None:
            error_by_lat = np.mean(error, axis=1)
            ax2.plot(lat_grid, error_by_lat)
            ax2.set_xlabel("Latitude")
        else:
            error_by_lat = np.mean(error, axis=1)
            ax2.plot(error_by_lat)
            ax2.set_xlabel("Latitude Index")
        ax2.set_ylabel("Mean Absolute Error")
        ax2.set_title("Error vs Latitude")
        
        plt.tight_layout()
        return fig, (ax1, ax2)
