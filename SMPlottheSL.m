cd('C:\Users\admin\Documents\GitHub\GradProject')
clear all;
close all;
restoredefaultpath
addpath 'C:\Users\admin\Documents\Run_Matlab_Fast_Folder'
Main_dir_hazirim= 'C:\Users\admin\Documents\Run_Matlab_Fast_Folder\Tfas_log_triag\Drives';
cd(Main_dir_hazirim)

addpath ('C:\Users\admin\Documents\GitHub\GradProject\mi');
addpath('C:\Users\admin\Documents\GitHub\GradProject\github_repo');
addpath('C:\Users\admin\Documents\GitHub\GradProject\rp-master\rp-master');
 addpath ('C:\Users\admin\Documents\GitHub\GradProject');
 addpath ('C:\Users\admin\Documents\GitHub\GradProject\knee_pt');
addpath 'C:\Users\admin\Documents\GitHub\GradProject\Beast\Matlab';
addpath 'C:\Users\admin\Documents\GitHub\GradProject\partitions';
%for drive 1, energy 4

drive_vec=0.6:0.2:2.8;
 drive_vec=[0.6 1 1.6 2.2 2.8];
  drive_vec=[ 1.6];
Mfit_vec=log(8.3*(10^3).*exp(-2.9.*drive_vec)+4.6*10^2);
std_fit=Mfit_vec-log((8.3*(10^3).*exp(-2.9.*drive_vec)+4.6*10^2)-(3.1*(10^3).*exp(-2.2.*drive_vec)+2.5*10^2));
std_fit2=Mfit_vec-log((8.3*(10^3).*exp(-2.9.*drive_vec)+4.6*10^2)-(3.1*(10^3).*exp(-2.2.*drive_vec)+2.5*10^2));
%inf_vec=zeros(1,length(drive_vec));
inf_vec1=zeros(1,length(drive_vec));
inf_vec2=zeros(1,length(drive_vec));
R_vec=zeros(1,length(drive_vec));
coordinate_number=2;
folderName=horzcat(' Dimension ',num2str(coordinate_number),' Run ', strrep(datestr(datetime), ':', '_'));
mkdir(folderName);
 cd(folderName);
R_vec=zeros(1,length(drive_vec));
for kok=1:1:length(drive_vec)
coordinate_number=2;
if coordinate_number==2
coordinate_number2=2;
coordinate_number=3;
else 
coordinate_number2=coordinate_number;
end
    cd(Main_dir_hazirim)
    mu=drive_vec(kok);
Mydata=importdata(horzcat('A3_reduced_PCA_',regexprep(num2str(mu, '%5.1f'),'\.','_'),'.mat'));
I=find(Mydata(:,coordinate_number+1)~=0);
mappx=Mydata(I,:);
mappx=[ mappx(:,1:3) log(mappx(:,4))];% the log thing
Mfit_vec(kok)=median(mappx(:,4)); %so its the median value and not the mean

% %Michael add according to Gili's request 12/2/23, add the distribution of
% %remaining time to FAS
% cd('C:\Users\admin\Pictures');
% n=1000; minimum_val=min(mappx(:,4)); maximum_val=max(mappx(:,4)); xw=mappx(:,4);
% pts=linspace(minimum_val,maximum_val,n);    [f,xi] = ksdensity(xw,pts,'Function','pdf');
% % 
% % gg=figure; 
% %   bbb=get(gg,'Position');
% %   new_width=8.7;
% %   set(gg, 'Units', 'centimeters', 'Position',[0 0 new_width 1*new_width]);
% % plot(xi,f);
% % hold on;
% % mean_mean=mean(xw);
% % xline(mean_mean);
% % hold on;
% % medi_medi=median(xw);
% % xline(medi_medi,'--');
% % xlim([minimum_val; maximum_val]);
% % xlabel('$Y$','FontSize',6,'Interpreter','latex');
% % rr=title(horzcat('$\Delta \mu = ',num2str(mu, '%5.1f'),'$\ $[K_{B}T]$'),'FontSize',6,'interpreter','latex');
% % set(gca,'FontSize',6);
% % legend('$f(Y)$','$\langle Y \rangle$','$M(Y)$','Interpreter','latex','FontSize',6,'Location','northwest');
% % print (horzcat('SIMedvsMean_', regexprep(num2str(mu, '%5.1f'),'\.','_'),'a'),'-dpng','-r600');
% % close(gg);
% 
%     cd(Main_dir_hazirim);
%Now check median or not

yo =sort(mappx(:,4));
aop=ceil(0.16*length(yo));
std_fit(kok)= median(mappx(:,4))-yo(aop);
%first randomizion
    x=mappx;
    random_x = x(randperm(size(x, 1)), :);
    mappx=random_x;
mindex=floor(0.6*size(mappx,1));% the index for the size of learning
top_data=floor(size(mappx,1)*0.8);
iter_num=10;% the number of cross validation iterations, on size of learning og mindex
tfas_reference=1;% The mu and std of the original histogram is:mu=1*10^7, or 1000 in coarsed-grained method, std=1*10^7,or 1000 in coarsed-grained method
CM = jet((iter_num));
red=length(1:mindex); %red is size learning set 
size_validation_set=length(mindex:top_data);
%mega matrix of tfas
tfas_predict_mat=zeros(iter_num,size_validation_set);
tfas_actually_mat=zeros(iter_num,size_validation_set);


mean_error_mat=zeros(iter_num,size_validation_set);
if coordinate_number2==2
    coordinate_number=2;
end

    %5D
    mini1=(min(mappx(:,1)));
    maxi1=(max(mappx(:,1)));
    mini2=(min(mappx(:,2)));
    maxi2=(max(mappx(:,2)));   
    mini3=(min(mappx(:,3)));
    maxi3=(max(mappx(:,3))); 
    if coordinate_number==5
    mini4=(min(mappx(:,4)));
    maxi4=(max(mappx(:,4)));   
    mini5=(min(mappx(:,5)));
    maxi5=(max(mappx(:,5))); 
    end
    d1=linspace(floor(mini1),ceil(maxi1),150);
    d2=linspace(floor(mini2),ceil(maxi2),150);
    d3=linspace(floor(mini3),ceil(maxi3),150);
      if coordinate_number==5
    d4=linspace(floor(mini4),ceil(maxi4),10);
    d5=linspace(floor(mini5),ceil(maxi5),10);
    [x0,y0,z0,w0,v0] = ndgrid(d1,d2,d3,d4,d5);

    XI = [x0(:) y0(:) z0(:) w0(:) v0(:)];
      else 

    [x0,y0,z0] = ndgrid(d1,d2,d3);
        XI = [x0(:) y0(:) z0(:) ];  
      end  




for zz=1:1:iter_num

    x=mappx;
    random_x = x(randperm(size(x, 1)), :);
    mappx=random_x;
    mip=mappx(mindex:top_data,:); %validation set
    mop=mappx(1:(mindex-1),:); %learning set

%     %2D option
% coordinate_number=2;
%     mini1=(min(mop(:,1)));
%     maxi1=(max(mop(:,1)));
%     mini2=(min(mop(:,2)));
%     maxi2=(max(mop(:,2)));   
% 
%     d1=linspace(floor(mini1),ceil(maxi1),33);
%     d2=linspace(floor(mini2),ceil(maxi2),33);
%     [x0,y0] = ndgrid(d1,d2);
%     X=[mop(:,1) mop(:,2)];
%     Y=mop(:,4);
%     XI = [x0(:) y0(:)];
%     YI = griddatan(X,Y,XI);
%     YI = reshape(YI, size(x0));
%     Ix= discretize(mip(:,1),d1);  
%     Iy= discretize(mip(:,2),d2);  
%     meanx=mean(Ix(~isnan(Ix)));
%     Ix(isnan(Ix))=round(meanx);
%     meany=mean(Iy(~isnan(Iy)));
%     Iy(isnan(Iy))=round(meany);

%     %3D
%     mini1=(min(mop(:,1)));
%     maxi1=(max(mop(:,1)));
%     mini2=(min(mop(:,2)));
%     maxi2=(max(mop(:,2)));   
%     mini3=(min(mop(:,3)));
%     maxi3=(max(mop(:,3))); 
% 
%     d1=linspace(floor(mini1),ceil(maxi1),33);
%     d2=linspace(floor(mini2),ceil(maxi2),33);
%     d3=linspace(floor(mini3),ceil(maxi3),33);
%     [x0,y0,z0] = ndgrid(d1,d2,d3);
%     X=mop(:,1:3);
%     Y=mop(:,4);
%     XI = [x0(:) y0(:) z0(:)];
%     YI = griddatan(X,Y,XI);
%     YI = reshape(YI, size(x0));
%     Ix= discretize(mip(:,1),d1);  
%     Iy= discretize(mip(:,2),d2);  
%     Iz= discretize(mip(:,3),d3);  
%     meanx=mean(Ix(~isnan(Ix)));
%     Ix(isnan(Ix))=round(meanx);
%     meany=mean(Iy(~isnan(Iy)));
%     Iy(isnan(Iy))=round(meany);
%     meanz=mean(Iz(~isnan(Iz)));
%     Iz(isnan(Iz))=round(meanz);

    %5D
    mini1=(min(mop(:,1)));
    maxi1=(max(mop(:,1)));
    mini2=(min(mop(:,2)));
    maxi2=(max(mop(:,2)));   
    mini3=(min(mop(:,3)));
    maxi3=(max(mop(:,3))); 
    if coordinate_number==5
    mini4=(min(mop(:,4)));
    maxi4=(max(mop(:,4)));   
    mini5=(min(mop(:,5)));
    maxi5=(max(mop(:,5))); 
    end
    d1=linspace((mini1),(maxi1),150);
    d2=linspace((mini2),(maxi2),150);
    d3=linspace((mini3),(maxi3),150);
      if coordinate_number==5
        d4=linspace(floor(mini4),ceil(maxi4),10);
        d5=linspace(floor(mini5),ceil(maxi5),10);
        [x0,y0,z0,w0,v0] = ndgrid(d1,d2,d3,d4,d5);
        X=mop(:,1:5);
        Y=mop(:,6);
        XI = [x0(:) y0(:) z0(:) w0(:) v0(:)];
      elseif  coordinate_number==3
        [x0,y0,z0] = ndgrid(d1,d2,d3);
        X=mop(:,1:3);
        Y=mop(:,4);
        XI = [x0(:) y0(:) z0(:) ]; 
     elseif  coordinate_number==2

        [y0,x0] = ndgrid(d1,d2);
        X=mop(:,1:2);
        Y=mop(:,4);
        XI = [y0(:) x0(:)  ]; 
      end
    YI = griddatan(X,Y,XI);
%     YI = griddatan(XI,YI,XI);
%     YI = griddatan(XI,YI,XI);
%     YI = griddatan(X,Y,XI,'nearest');
%Michael add to kill the smoothing 5/12/2022, just comment out the YI
     YI = reshape(YI, size(x0));
     %     YIi = reshape(YIi, size(x0));

     intergal_dist=2;
k=ones(intergal_dist)/(intergal_dist*intergal_dist-1);k(ceil(intergal_dist/2),ceil(intergal_dist/2))=0;
averageIntensities = conv2(double(YI),k,'same');
YI =averageIntensities;%this is 10 times with 3 k
     intergal_dist=2;
k=ones(intergal_dist)/(intergal_dist*intergal_dist-1);k(ceil(intergal_dist/2),ceil(intergal_dist/2))=0;
averageIntensities = conv2(double(YI),k,'same');
YI =averageIntensities;%this is 10 times with 3 k


    Ix= discretize(mip(:,1),d1);  
    Iy= discretize(mip(:,2),d2);  
    Iz= discretize(mip(:,3),d3); 
          if coordinate_number==5
            Iw= discretize(mip(:,4),d4);  
            Iv= discretize(mip(:,5),d5);  
          end


    meanx=mean(Ix(~isnan(Ix)));
    Ix(isnan(Ix))=round(meanx);
    meany=mean(Iy(~isnan(Iy)));
    Iy(isnan(Iy))=round(meany);
    meanz=mean(Iz(~isnan(Iz)));
    Iz(isnan(Iz))=round(meanz);
              if coordinate_number==5

    meanw=mean(Iw(~isnan(Iw)));
    Iw(isnan(Iw))=round(meanw);
    meanv=mean(Iv(~isnan(Iv)));
    Iv(isnan(Iv))=round(meanv);
              end
                    TFAS_real=zeros(size(mip,1),1);
                    TFAS_predicted=zeros(size(mip,1),1);
    
                    for ii=1:1:size(mip,1)

                        if coordinate_number==3
                        TFAS_real(ii)=mip(ii,coordinate_number+1);
                        TFAS_predicted(ii)=YI(Ix(ii),Iy(ii),Iz(ii));
                        elseif coordinate_number==2
                        TFAS_real(ii)=mip(ii,coordinate_number+2);
                        TFAS_predicted(ii)=YI(Ix(ii),Iy(ii));
                        elseif coordinate_number==5
                        TFAS_real(ii)=mip(ii,coordinate_number+1);
                        TFAS_predicted(ii)=YI(Ix(ii),Iy(ii),Iz(ii),Iw(ii),Iv(ii));
                        end
                        if isnan(TFAS_predicted(ii))
                             TFAS_predicted(ii)=mean(Y);
                        end
                    end


                    tfas_predict_mat(zz,1:end)=TFAS_predicted;
                    tfas_actually_mat(zz,1:end)=TFAS_real;
                    mean_error_mat(zz,1:end)=abs(TFAS_real-TFAS_predicted);

end
%  tfas_predict_mat=exp( tfas_predict_mat);
%  tfas_actually_mat= exp(tfas_actually_mat);
%mkdir(folderName);
% cd(folderName)



            
%             Sapphire=zeros(length(iter_num),1);
% 
%             for zz=1:1:iter_num
%  
%                 look=abs(tfas_predict_mat(zz,1:end)-tfas_actually_mat(zz,1:end)).^2;
%                 Sapphire(zz)=mean(look);
% 
%             end
%            ruby=sqrt(mean(Sapphire));
%      
%    
% overall_minimal_mean_error=ruby;

        nn=figure;
        fullscreen();     

              xx=squeeze(tfas_predict_mat(1,1:end));
            [x,I]=sort(xx);
        bin_width=0.5;
        smooth_win=0;
        ticking_bomb=3.5:0.5:7.5;
        ticking_bomby=3:2:9;
min_of_all=min(min(tfas_predict_mat));
max_of_all=max(max(tfas_predict_mat));
%         hista=linspace(bin_width*floor(min_of_all/bin_width),bin_width.*ceil(max_of_all/bin_width),floor((max_of_all-min_of_all)/bin_width)+2);
hista=linspace(bin_width*floor(min_of_all/bin_width),bin_width.*ceil(max_of_all/bin_width),(ceil(max_of_all/bin_width)-floor(min_of_all/bin_width)+1));
          
  meaning=zeros(iter_num,length(hista)-1);
            stding=zeros(iter_num,length(hista)-1);

            for zz=1:1:iter_num          
            xx=squeeze(tfas_predict_mat(zz,1:end));
            [x,I]=sort(xx);
%             yy=squeeze(tfas_actually_mat(zz,1:end)-tfas_predict_mat(zz,1:end));
            yy=squeeze(tfas_actually_mat(zz,1:end));
            y=yy(I);
            subplot(4,1,1)
            scatter((x),(y),'filled','MarkerEdgeColor',CM(zz,:),  'MarkerFaceColor',CM(zz,:));
           % xticks(ticking_bomb(2:end));
            set(gca,'XTickLabel',[]);
            yticks(ticking_bomby);
            %ylabel('Log T_{FAS} First Subdivision')
                    ylabel({'$\hat{Y}$',' '},'Interpreter','latex');
                            xlim([ticking_bomb(1) ticking_bomb(end)]);
        ylim([ticking_bomby(1) ticking_bomby(end)]);
                            set(gca,'FontSize',24) ;
                                    set(gca,'box','off');
                                    set(gca,'xtick',[])

%             Y=[Y y];
            % X = [ones(length(x),1) x];
            % X=x;
%             hold on;
%                 b = x\(y);
%             yCalc2 = x*b;
%             plot(x,yCalc2,'--')
%             title('Scattered Data')
            y_new=y;

%         hista=linspace(bin_width*floor(min(x/bin_width)),bin_width*ceil(max(x)/bin_width),floor((max(x)-min(x))/bin_width)+2);
        std_hista=zeros(1,length(hista)-1);
        mean_hista=zeros(1,length(hista)-1);

        for ii=1:1:(length(hista)-1)
            I=find (hista(ii)-smooth_win<x & x<(hista(ii+1))+smooth_win);
            if length(I)<floor(0.01*length(x)) %do nto consider when there is not enough statistics,michael add
            std_hista(ii)=nan;
            mean_hista(ii)=nan;
            else
               yo =sort(y_new(I));
        mean_hista(ii)=median(y_new(I));
        %taking the lower std
        aop=ceil(0.16*length(yo));

        std_hista(ii)=(mean_hista(ii))-(yo(aop));
%         std_hista(ii)=std(y_new(I));

            end
        
        end
       x_hista=(hista(2:end)+hista(1:(end-1)))/2;
        
        I9=find (~isnan(mean_hista) &  x_hista>0,1);
        I2=find (isnan(mean_hista) &  x_hista>0,1);
        I3=find (isnan(std_hista) &  x_hista>0,1);
        I4=min(I2,I3);
        if isempty(I4)
          I4=length(mean_hista);  

        end
        if I9<I4
        I4=min(I2,I3);
        if isempty(I4)
          I4=length(mean_hista);  

        end
        else
        I2=find (isnan(mean_hista) &  x_hista>0,I9);
        I3=find (isnan(std_hista) &  x_hista>0,I9);
        I4=min(I2(end),I3(end));
        end
I4=length(mean_hista);
%michael unsure fix 
     %here michael switch to log scale the std. 27/11/22
    nn2=subplot(4,1,2);  
        %h2=plot((x_hista(1:I4)),(std_hista(1:I4)),'y--');
        %hold on;
        %h1=plot((x_hista(1:I4)),(mean_hista(1:I4)),'Color',CM(zz,:),'LineStyle','--');
        h1=scatter((x_hista(1:I4)),(mean_hista(1:I4)),'filled','Marker','square','MarkerEdgeColor',CM(zz,:),  'MarkerFaceColor',CM(zz,:));
        hold on;
        %h3=plot((x_hista(1:I4)),(x_hista(1:I4)),'k');
        
        meaning(zz,1:(length(hista)-1))=mean_hista;
        stding(zz,1:(length(hista)-1))=std_hista;
%         legend('Mean','STD','Linear Compare')
%         title('Predicted Perfromace')
%         legend('STD','Mean','Linear Compare')

       [mmmm,I5]= min(abs(x-x_hista(I4)));
        nn1=subplot(4,1,1);
         scatter((x(1:I5)),(y(1:I5)),'filled','MarkerEdgeColor',CM(zz,:),  'MarkerFaceColor',CM(zz,:));
            hold on;
                    if zz==1
                        %                     xlabel('T_{FAS} Predicted')
        %ylabel('Log T_{FAS} Measured All Subdivisions')
                ylabel({'${Y}_{CV}$',' '},'Interpreter','latex')
                           % xticks(ticking_bomb(2:end));
                           set(gca,'XTickLabel',[]);
                                        yticks(ticking_bomby);
                                                xlim([ticking_bomb(1) ticking_bomb(end)]);
        ylim([ticking_bomby(1) ticking_bomby(end)]);
                                                set(gca,'FontSize',24);
                                                        set(gca,'box','off');
                                                        set(gca,'xtick',[])
                    end

%          hold on;
%                 b = x\(y);
%             yCalc2 = x*b;
%             plot(x,yCalc2,'--')
%             title('Scattered Data')
            end
        mean_vec=zeros(1,size(mean_hista,2));
        std_vec=zeros(1,size(mean_hista,2));
        meaning(meaning==0)=nan;
        stding (stding==0)=nan;
        for row=1:1:size(meaning,2)
            uyu=meaning(:,row);
            utu=stding(:,row);
            nani=~isnan(uyu);
            uyu=uyu(nani);
            utu=utu(nani);
     %       if length(nani)>(iter_num/2) && sum(~isnan(uyu))>1, michael
     %       changed for sake of loks 12/1/2/22
            if  sum(~isnan(uyu))>0 
                mean_vec(row)=mean(uyu); 
            std_vec(row)=mean(utu);
            else
            mean_vec(row)=nan;  
            std_vec(row)=nan;
            end

        end
        std_vec2=std_vec;
        nn2=subplot(4,1,2);
        h3=plot((x_hista(1:I4)),(x_hista(1:I4)),'k--');
        hold on;
        %h4=plot((x_hista(1:end)),(mean_vec(1:end)),'r','MarkerSize',20);
        h4=errorbar((x_hista(1:end)),(mean_vec(1:end)),std_vec(1:end),std_vec2(1:end),'k');
           % xticks(ticking_bomb(2:end));
            set(gca,'XTickLabel',[]);
                        yticks(ticking_bomby);
                                xlim([ticking_bomb(1) ticking_bomb(end)]);
        ylim([ticking_bomby(1) ticking_bomby(end)]);
                set(gca,'box','off')
set(gca,'xtick',[])
                    set(gca,'FontSize',24) 
        %hold on;
        %h5=plot((x_hista(1:end)),(std_vec(1:end)),'b','MarkerSize',20);
                 %   ylim([2 log(2000)])

%         xlabel('T_{FAS} Predicted')
      %  ylabel('Log T_{FAS} Cross Validation')
              ylabel({'$<{Y}_{CV}>$',' '},'Interpreter','latex'); 
              xh = get(gca,'ylabel'); % handle to the label object
p = get(xh,'position'); % get the current position property
%p(2) = 2*p(2) ;   
%      legend([h1 h2 h3 h4 h5],'Mean','STD','Linear Compare','Iterations mean','Iterations STD')
%         sgtitle(horzcat('Weights Number ',num2str(mm)))

%Michael adds the remaining 10%

CV_offset=(mean_vec)-(x_hista);
mip=mappx(top_data:end,:);%reaaning check set
size_mvp=length(mip);
tfas_predict_mat2=zeros(1,size_mvp);
tfas_actually_mat2=zeros(1,size_mvp);
mean_error_mat2=zeros(1,size_mvp);
%study on a randomized 60%
  x=mappx;
  random_x = x(randperm(size(x, 1)), :);
   mappx=random_x;
%     mip=mappx(mindex:top_data,:); %validation set
  mop=mappx(1:(mindex-1),:); %learning set

    %5D
    mini1=(min(mop(:,1)));
    maxi1=(max(mop(:,1)));
    mini2=(min(mop(:,2)));
    maxi2=(max(mop(:,2)));   
    mini3=(min(mop(:,3)));
    maxi3=(max(mop(:,3))); 
    if coordinate_number==5
    mini4=(min(mop(:,4)));
    maxi4=(max(mop(:,4)));   
    mini5=(min(mop(:,5)));
    maxi5=(max(mop(:,5))); 
    end
    d1=linspace((mini1),(maxi1),150);
    d2=linspace((mini2),(maxi2),150);
    d3=linspace((mini3),(maxi3),150);
      if coordinate_number==5
        d4=linspace(floor(mini4),ceil(maxi4),10);
        d5=linspace(floor(mini5),ceil(maxi5),10);
        [x0,y0,z0,w0,v0] = ndgrid(d1,d2,d3,d4,d5);
        X=mop(:,1:5);
        Y=mop(:,6);
        XI = [x0(:) y0(:) z0(:) w0(:) v0(:)];
      elseif  coordinate_number==3
        [x0,y0,z0] = ndgrid(d1,d2,d3);
        X=mop(:,1:3);
        Y=mop(:,4);
        XI = [x0(:) y0(:) z0(:) ]; 
     elseif  coordinate_number==2

        [y0,x0] = ndgrid(d1,d2);
        X=mop(:,1:2);
        Y=mop(:,4);
        XI = [y0(:) x0(:)  ]; 
      end
    YI = griddatan(X,Y,XI);
%     YI = griddatan(XI,YI,XI);
%     YI = griddatan(XI,YI,XI);
%     YI = griddatan(X,Y,XI,'nearest');
     YI = reshape(YI, size(x0));
     %Michael kill the smoothing 5/12/22
     intergal_dist=2;
k=ones(intergal_dist)/(intergal_dist*intergal_dist-1);k(ceil(intergal_dist/2),ceil(intergal_dist/2))=0;
averageIntensities = conv2(double(YI),k,'same');
YI =averageIntensities;%this is 10 times with 3 k
     intergal_dist=2;
k=ones(intergal_dist)/(intergal_dist*intergal_dist-1);k(ceil(intergal_dist/2),ceil(intergal_dist/2))=0;
averageIntensities = conv2(double(YI),k,'same');
YI =averageIntensities;%this is 10 times with 3 k

%Z(isnan(Z))=log(5*10^3);
% gg=figure; 
%   bbb=get(gg,'Position');
%   h_factor=bbb(3)/bbb(4);
%   new_width=8.7;
%   set(gg, 'Units', 'centimeters', 'Position',[2 2 new_width 1*new_width]);
% surf(x0,y0,YI,'edgecolor','none');
% xox=xlabel('$y_{1}$','FontSize',6,'interpreter','latex');
% % xox.Position=[0.245618897354873,-2.687465559516938,3.125798572273085];
% xox.Position=[-3.028814764086392,1.358019974973075,2.797098988652167];
% yo=ylabel('$y_{2}$','FontSize',6,'interpreter','latex');
% % yo.Position=[-3.046650439611242,1.723087691146219,3.178316238429432];
% yo.Position=[0.206455320397623,-3.039637698293576,2.693752328886021];
% zozz=zlabel('$\hat{Y}_{p}$','FontSize',6,'interpreter','latex','Rotation',0);
% % zozz.Position=[-2.969896712203863,5.565259539735991,5.803416416230605];
% zozz.Position=[-2.838864659216853,5.059708293482914,5.396784659519299];
% 
% rr=title(horzcat('$\Delta \mu = ',num2str(mu),'$\ $[K_{B}T]$'),'FontSize',6,'interpreter','latex');
% set(gca,'FontSize',6);
% cd('C:\Users\admin\Pictures');
% saveas(gg,horzcat('SMSL' ,'.fig'));
% % print ('StressY','-depsc','-r600');
% print ('SMSL','-dpng','-r600');

gg=figure; 
  bbb=get(gg,'Position');
  h_factor=bbb(3)/bbb(4);
  new_width=8.7;
  set(gg, 'Units', 'centimeters', 'Position',[2 2 new_width 1*new_width]);
  YI(isnan(YI))=log(5*10^3);
imagesc(x0(:),y0(:),YI);
xox=xlabel('$y_{1}$','FontSize',6,'interpreter','latex');
% xox.Position=[0.245618897354873,-2.687465559516938,3.125798572273085];
% xox.Position=[-3.028814764086392,1.358019974973075,2.797098988652167];
yo=ylabel('$y_{2}$','FontSize',6,'interpreter','latex','Rotation',0);
% yo.Position=[-3.046650439611242,1.723087691146219,3.178316238429432];
% yo.Position=[0.206455320397623,-3.039637698293576,2.693752328886021];
% zozz=zlabel('$\hat{Y}_{p}$','FontSize',6,'interpreter','latex','Rotation',0);
colorbar()
% zozz.Position=[-2.969896712203863,5.565259539735991,5.803416416230605];
% zozz.Position=[-2.838864659216853,5.059708293482914,5.396784659519299];

rr=title(horzcat('$\Delta \mu = ',num2str(mu),'$\ $ K_{B}T$'),'FontSize',6,'interpreter','latex');
set(gca,'FontSize',6);
cd('C:\Users\admin\Pictures');
saveas(gg,horzcat('SMSL' ,'.fig'));
% print ('StressY','-depsc','-r600');
print ('SMSL','-dpng','-r600');


end