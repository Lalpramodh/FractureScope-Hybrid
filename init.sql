CREATE TABLE predictions (
    id INT AUTO_INCREMENT PRIMARY KEY,
    username VARCHAR(100),
    image_path VARCHAR(255),
    prediction VARCHAR(100),
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
);
