import flash
import torch
from flash.tabular import TabularClassificationData, TabularClassifier

# 1. Create the DataModule
datamodule = TabularClassificationData.from_csv(
    categorical_fields=["Sex", "Age", "SibSp", "Parch", "Ticket", "Cabin", "Embarked"],
    numerical_fields="Fare",
    target_fields="Survived",
    train_file="https://pl-flash-data.s3.amazonaws.com/titanic.csv",
    val_split=0.1,
    batch_size=8,
)

# 2. Build the task
model = TabularClassifier.from_data(datamodule, backbone="fttransformer")

# 3. Create the trainer and train the model
trainer = flash.Trainer(max_epochs=3, accelerator='gpu', devices=[2])
trainer.fit(model, datamodule=datamodule)

# 4. Generate predictions from a CSV
datamodule = TabularClassificationData.from_csv(
    predict_file="https://pl-flash-data.s3.amazonaws.com/titanic.csv",
    parameters=datamodule.parameters,
    batch_size=64,
)
predictions = trainer.predict(model, datamodule=datamodule, output="classes")
print(predictions)

# 5. Save the model!
trainer.save_checkpoint("tabular_classification_model.pt")