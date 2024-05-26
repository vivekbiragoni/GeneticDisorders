# steps to update your submodule in the parent repository:

1. **Navigate to the parent repository**:
   ```sh
   C:\Users\vivek\Documents\GitHub\Genetic-Disorders
   ```

2. **Navigate to the submodule directory**:
   ```sh
   cd .\Yolov8_child_repo\
   ```

3. **Pull the latest changes from the submodule repository**:
   ```sh
   git pull origin main
   ```

4. **Navigate back to the parent repository**:
   ```sh
   cd ..
   ```

5. **Update the submodule reference and push changes**:
   ```sh
    git add .\Yolov8_child_repo\
   git commit -m "commit message"
   git push
   ```
