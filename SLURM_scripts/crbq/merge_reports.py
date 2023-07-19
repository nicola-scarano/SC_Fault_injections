import argparse
import pandas as pd
import os
import sys
from multiprocessing import Pool

def merge_fsim_reports(path):
    pd.set_option('mode.use_inf_as_na', True)   
    if path:
        items = os.listdir(path)
        csvfiles = [item for item in items if "F_" in item]  
        if len(csvfiles)>0:
            if (os.path.exists(os.path.join(path,"Faulty_boxes_report.csv"))):
                for rep_per_fault in csvfiles:
                    os.remove(os.path.join(path,rep_per_fault))
            else:
                full_report = pd.DataFrame()
                for rep_per_fault in csvfiles:
                    fault_list= pd.read_csv(os.path.join(path,rep_per_fault),index_col=[0]) 
                    full_report=pd.concat([full_report,fault_list],axis=0, ignore_index=True, sort=False)                
                full_report.to_csv(os.path.join(path,"Faulty_boxes_report.csv"))   

                for rep_per_fault in csvfiles:
                    os.remove(os.path.join(path,rep_per_fault))         
        
        fault_list_file=os.path.join(path,"fault_list.csv")
        fsim_report_file=os.path.join(path,"fsim_report.csv")
        fault_list= pd.read_csv(fault_list_file,index_col=[0]) 
        fsim_report= pd.read_csv(fsim_report_file,index_col=[0]) 

        # index=((fsim_report['gold_ACC@1'].isna()==False) | (fsim_report['gold_ACC@k'].isna()==False))
        # index=(fsim_report['gold_iou@1'].isna()==False)
        # fsim_report=fsim_report.loc[index]
        full_reportfs=pd.concat([fault_list,fsim_report],axis=1)
        full_reportfs.to_csv(os.path.join(path,"fsim_full_report.csv"))

        flist= fault_list 
        df_pivot = pd.read_csv(os.path.join(path,"Faulty_boxes_report.csv"),index_col=[0]) 
        df_pivot = df_pivot.pivot_table(index=['FaultID', 'imID'], columns='Pred_idx', 
                          values=['G_lab', 'F_lab','iou score', 'area_ratio','f_candidate_conf', 'G_score'])
        df_pivot.columns = [f'{col}{int(idx)}' for col, idx in df_pivot.columns]
        df_pivot = df_pivot.reset_index()
        
        #df_pivot = df_pivot[['FaultID','imID','Pred_idx','G_pred','F_pred','G_clas','F_clas','G_Target']]
        #df_pivot.to_csv(os.path.join(args.path,"Faulty_boxes_report.csv")) 
        
        FaultID=[f"F_{item}_results" for item in range(len(flist))]
        FaultID_col=pd.DataFrame({"FaultID":FaultID})
        nflist=pd.concat([FaultID_col,flist],axis=1)
        left=nflist.set_index('FaultID')
        right=df_pivot.set_index(['FaultID','imID'])
        result = pd.merge(
            left.reset_index(), right.reset_index(), on=["FaultID"], how="inner"
        ).set_index(["FaultID","imID"])
        result=result.reset_index()
        result.to_csv(os.path.join(path,"Faulty_boxes_report.csv"))
        print(f"merged {path}")


def main(args):
    path=args.path
    workers=args.workers
    items = os.listdir(path)
    directories=[item for item in items if os.path.isdir(os.path.join(path,item))]
    pool_directories=[]
    for directory in directories:
        merge_files_path=os.path.join(path,directory)
        # list_items = os.listdir(newdir)
        # sim_dir=[item for item in list_items if os.path.isdir(os.path.join(newdir,item))]
        # merge_files_path=os.path.join(newdir,sim_dir[0])
        pool_directories.append(merge_files_path)
    with Pool(processes=workers) as pool:
        results = pool.map_async(merge_fsim_reports, pool_directories) 
        results.wait()
    output = results.get()

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='pytorchFI report merge')
    parser.add_argument('-f', '--path', metavar='DIR',
                        help='path to dataset')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
    print("start conversion...")
    main(parser.parse_args())
    print("finish conversion...")