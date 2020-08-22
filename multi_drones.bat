@SET "d=Drone"
@setlocal EnableDelayedExpansion
@SET cx[1]=0
@SET cx[2]=0
@SET cy[1]=1
@SET cy[2]=-1

@for /L %%x in (1, 1, %1) do @(
   echo %%x
   start "" C:\Users\khizar\anaconda3\Scripts\activate.bat ^&^& conda activate unreal ^&^& python oorbit.py --name "%d%%%x" --radius 4 --altitude 2 --cx %%cx[%%x]%% --cy %%cy[%%x]%% --iterations 2 --snapshots 20 ^&^& exit
)