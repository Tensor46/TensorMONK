### essentials
* CudaModel: Converts any model (pytorch module/sequential) to run on single/multiple gpu or cpu
* LoadModel: Loads pre-trained models with CudaModel
* MakeModel: Builds model using base class
* SaveModel: Save models (state_dict of anything in CudaModel (nn.Module) that starts with net in BaseModel, and rest as is)
* EasyTrainer: Similar to pytorch Trainer with more flexibility to train any neural architecture.
