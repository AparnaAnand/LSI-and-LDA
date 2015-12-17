
#Edited on Saturday
type ti_value
	tf::Float64
	idf::Float64
	tfidf::Float64
end

#-----------------------------#
#Read and pre-process data

docs={}

start_dir=pwd()
current_dir="../DataSet/20Newsgroups"
cd(current_dir)

function goInto(dir)
	for file in readdir(dir)
		if isdir(file)
			cd(file)
			goInto(dir*"/"*file)
			cd(dir)
		else 
			if file[1]!='.'
				push!(docs,abspath(file))
			end
		end
	end
end

goInto(current_dir)
cd(start_dir)

noOfDocs=length(docs)


f=open("../DataSet/StopWords/common-english-words.txt")
stopWords=split(readall(f),',')
append!(stopWords,["from:","subject:","organization:","distribution:","lines:","re:",""])

println("Processed Stop Words")

all_words={}
word_doc=Array(Vector{String}, noOfDocs)
newWords={}
new={}
punctuations=['.',',',':',';','?','!','<','>','(',')','{','}','-',' ','\n','\r','"','_','/','*']
for i=1:noOfDocs
	f=open(docs[i])
	s=lowercase(readall(f))
	newWords=split(s,punctuations)
	append!(new,newWords)
	for j=1:length(new)
		p = match(r"^\s*(.*)\s*(.*)$", new[j])
		new[j]=p.captures[1]
		append!(new,[convert(ASCIIString,p.captures[2])])
	end
	for j=1:length(new)
		p = match(r"^[^A-Za-z0-9]?(.*)[^A-Za-z0-9]?$", new[j])
		new[j]=p.captures[1]
	end
	for j=1:length(new)
		p = match(r"^[A-Za-z]*(-|')?[A-Za-z]*$", new[j])
		if p== nothing
			new[j]=""
		end
	end
	delIndices=findin(new,stopWords)
	deleteat!(new,delIndices)
	word_doc[i]=new
	append!(all_words,new)
	splice!(new,1:length(new))
	close(f)
	println("Processed words of document $i")
end

all_words=unique(all_words)



println("Creating List of terms per Documents; Calculating TF")
term_doc=Array(ti_value,length(all_words),noOfDocs)

for i=1:noOfDocs
	maxOcc=0
	for j=1:length(all_words)
		ind=findin(word_doc[i],[all_words[j]]) 
		term_doc[j,i]=ti_value(length(ind),0,0)
		if maxOcc<length(ind)
			maxOcc=length(ind)
		end
	end
	#println("Max Occ: $maxOcc")
	for j=1:length(all_words)
		term_doc[j,i].tf=term_doc[j,i].tf/maxOcc
	end
	println("TF Processed Doc $i")
end







println("IDF Modified")

for i=1:length(all_words)
	count=0
	for j=1:noOfDocs
		if term_doc[i,j].tf>0
			count+=1
		end
	end
	idf_val=noOfDocs/count
	
	for j=1:noOfDocs
		term_doc[i,j].idf=log2(idf_val)
	end
	println("IDF processed Term $i")
end


println("Final Processing")
for i=1:noOfDocs
	for j=1:length(all_words)
		term_doc[j,i].tfidf=term_doc[j,i].tf*term_doc[j,i].idf
	end
end

println("Calculating TF-IDF")

tdoc=Array(Float64,length(all_words),noOfDocs)
for i=1:noOfDocs
	for j=1:length(all_words)
		tdoc[j,i]=term_doc[j,i].tfidf
	end
end


println("TF IDF Processed")



println("Processing SVD")

SVD=svdfact(tdoc)




dim=100

println(dim)

S=diagm(SVD[:S])
Vt=SVD[:Vt]

Sk=S[1:dim,1:dim]
Vtk=Vt[1:dim,:]

DocRep=*(Sk,Vtk)




#-----------------------------------#
type pt
	coords
	index
end


type clusters
	pts
	centroid::pt
end

#-----------------------------#

k=5
kpoints=Array(pt,k)
DataPoints= Array(pt,0)
ClusterList=Array(clusters,k)


println("Creating DataPoints")
#-----------------------------#

for i = 1:noOfDocs
	p=Array(Float64,dim)
	for j=1:dim
		p[j]=DocRep[j,i]
	end
	push!(DataPoints,pt(p,i))
end


println("DataPoints created.")

#-----------------------------#
#Preserve Original DataPoints for final reassignment

DataPointsOrg=deepcopy(DataPoints)




#-----------------------------#
#Find cosine similarity

function distance(p1,p2)
	#println("P1: $p1 \n P2: $p2")
	dotProd=dot(p1.coords,p2.coords)
	modP1=0
	modP2=0
	for i=1:dim
		modP1+=p1.coords[i]^2
		modP2+=p2.coords[i]^2
	end
	modP1=sqrt(modP1)
	modP2=sqrt(modP2)
	cosTheta=dotProd/(modP1*modP2)
	if cosTheta > 1
		cosTheta =1
	end	
	return acos(cosTheta)
end

#-----------------------------#

#Finding k farthest points
println("Calculating k farthest points")

indexOne=rand(1:length(DataPoints))
kpoints[1]=DataPoints[indexOne]
splice!(DataPoints,indexOne)


j=1
while(j<k)
	maxdist=0
	maxindex=0
	r=length(DataPoints)-j
	for x=1:r
		dist=0
		for y=1:j
			dist+=distance(kpoints[y],DataPoints[x])
		end
		if dist>maxdist
			maxdist=dist
			maxindex=x
		end
	end
	j+=1
	kpoints[j]=DataPoints[maxindex]	
	splice!(DataPoints,maxindex)
end

println("Found K farthest Points")
#println("KPOINTS\n",kpoints)




#-----------------------------#
#Assign the k points to k clusters

function assignkpoints()
	for i=1:k
		p=Array(pt,1)
		p[1]=kpoints[i]
		ClusterList[i]=clusters(p,kpoints[i])
	end
end


#-----------------------------#
#Function to recalculate centroid


#Function to recalculate centroid

function adjustCentroid(cluster)
	total=zeros(Array(Float64,dim))
	for i=1:length(cluster.pts)
		for j=1:dim
			total[j]+=cluster.pts[i].coords[j]
		end
	end
	for i=1:dim
		total[i]=total[i]/length(cluster.pts)
	end
		cluster.centroid=pt(total,0)
end


#-----------------------------#
#Function to assign each DataPoint to a cluster

function assign(p)
	minIndex=0
	minDist=10000000000000000
	for i=1:k
		dist=distance(p,ClusterList[i].centroid)
		if(dist<minDist)
			minDist=dist
			minIndex=i
		end
	end
	push!(ClusterList[minIndex].pts,p)
	#adjustCentroid(ClusterList[minIndex])
end




#-----------------------------#
println("Started Clustering")
#Start Clustering
trial=1
flag =1
assignkpoints()
while flag == 1
	println("Trial $trial begin")
	for i=1:length(DataPoints)
		assign(DataPoints[i])
	end
	for i=1:k
		adjustCentroid(ClusterList[i])
	end
	
	total=0
	for i=1:k
		total+=length(ClusterList[i].pts)
		println("Cluster $i: ",length(ClusterList[i].pts))
	end
	println("Total No of Points ",total)
	println("Trial $trial over")
	trial+=1

	oldKpoints=deepcopy(kpoints)
	
	for i=1:k	
			kpoints[i]=ClusterList[i].centroid
	end
	
			
	
	
	DataPoints=deepcopy(DataPointsOrg)
	flag=0
	for i=1:k
		diff=distance(oldKpoints[i],kpoints[i])
		if  diff > 1.0e-4
			println("ProLogos: $diff")
			flag=1
			ClusterList=Array(clusters,k)
	
			for i=1:k	
				p=Array(pt,0)
				ClusterList[i]=clusters(p,kpoints[i])
			end	
		end	
	end
	
end



#----------------------------#
println("\nEvaluation of clusters")

type indexList
	pts
end

indices=Array(indexList,0)

for i=1:k
	index=Array(Int32,0)
	for j=1:length(ClusterList[i].pts)
		push!(index,ClusterList[i].pts[j].index)
	end
	push!(indices,indexList(index))
end

AvgPurity=0.0

countVal=Array(Int64,k)
for i=1:k
	countVal=zeros(Array(Int64,k))
	for j=1:length(indices[i].pts)
		if indices[i].pts[j] in (1:319)
			countVal[1]+=1
		elseif indices[i].pts[j] in (320:715)
			countVal[2]+=1
		elseif indices[i].pts[j] in (716:1112)
			countVal[3]+=1
		elseif indices[i].pts[j] in (1113:1508)
			countVal[4]+=1
		end
			
	end
	println("Cluster $i:")
	println("No of documents in cluster: ",length(ClusterList[i].pts))
	x=sortperm(countVal)
	println(x)
	
	for a=1:k
		println("Folder $a($(countVal[a])): $(countVal[a]/length(ClusterList[i].pts)*100)")
	end
	purity=maximum(countVal)/length(ClusterList[i].pts)
	println("Purity: ",purity)
	AvgPurity+=purity*length(ClusterList[i].pts)
	println("\n")
end
println("Avg Purity: ",AvgPurity/length(DataPoints))