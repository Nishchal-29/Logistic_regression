import math
import random

class LogisticRegression:
    def __init__(self, learning_rate=0.01):
        self.weights = [0, 0, 0, 0]  # For CGPA, experience, projects, languages
        self.bias = 0
        self.lr = learning_rate
    
    def sigmoid(self, z):
        return 1 / (1 + math.exp(-z))
    
    def predict(self, features):
        z = sum(w * x for w, x in zip(self.weights, features)) + self.bias
        return self.sigmoid(z)
    
    def train(self, X, y, epochs=1000):
        m = len(X)
        
        for _ in range(epochs):
            dw = [0] * len(self.weights)
            db = 0
            
            # Compute gradients
            for i in range(m):
                prediction = self.predict(X[i])
                error = prediction - y[i]
                
                for j in range(len(self.weights)):
                    dw[j] += error * X[i][j]
                db += error
            
            # Update weights and bias
            for j in range(len(self.weights)):
                self.weights[j] -= (self.lr * dw[j]) / m
            self.bias -= (self.lr * db) / m

def generate_dataset(num_samples=1000):
    X = []
    y = []
    
    for _ in range(num_samples):
        # Generate random features
        cgpa = 6 + random.random() * 4  # CGPA between 6 and 10
        experience = random.random() * 8  # 0-8 years
        projects = random.randint(0, 10)  # 0-10 projects
        languages = random.randint(1, 8)  # 1-8 languages
        
        # Create decision rule
        score = (
            (cgpa / 10) * 0.4 + 
            (experience / 8) * 0.3 + 
            (projects / 10) * 0.2 + 
            (languages / 8) * 0.1
        )
        
        X.append([cgpa, experience, projects, languages])
        y.append(1 if score > 0.6 else 0)  # Threshold for eligibility
    
    return X, y

def main():
    # Generate training data
    print("Generating training data...")
    X_train, y_train = generate_dataset(1000)
    
    # Train model
    print("\nTraining model...")
    model = LogisticRegression(learning_rate=0.01)
    model.train(X_train, y_train, epochs=2000)
    
    # Interactive prediction loop
    print("\nHR Candidate Screening System")
    print("-----------------------------")
    
    while True:
        try:
            print("\nEnter candidate details (or press Ctrl+C to exit):")
            cgpa = float(input("CGPA (0-10): "))
            experience = float(input("Years of experience: "))
            projects = int(input("Number of projects: "))
            languages = int(input("Programming languages known: "))
            
            features = [cgpa, experience, projects, languages]
            probability = model.predict(features)
            
            print("\nPrediction Results:")
            print("-----------------")
            print(f"Eligibility Score: {probability:.2%}")
            print(f"Status: {'Eligible' if probability >= 0.5 else 'Not Eligible'} for interview")
            
        except KeyboardInterrupt:
            print("\n\nThank you for using the HR Screening System!")
            break
        except ValueError:
            print("\nError: Please enter valid numeric values!")

if __name__ == "__main__":
    main()