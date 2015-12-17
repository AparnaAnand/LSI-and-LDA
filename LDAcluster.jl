type pt
	coords
	index
end


type cluster
	pts
end

k=4
ClusterList=Array(cluster,k)
for i=1:k
	p=Array(Int32,0)
	ClusterList[i]=cluster(p)
end


for i=1:length(docs)
	max=0.0
	maxIndex=0
	for j=1:k
		if topic_dist[i][j].freq > max
			max=topic_dist[i][j].freq
			maxIndex=j
		end
	end
	push!(ClusterList[maxIndex].pts,i)
end
countVal=Array(Int64,k)


AvgPurity=0.0
for i=1:k
	countVal=zeros(Array(Int64,k))
	for j=1:length(ClusterList[i].pts)
		if ClusterList[i].pts[j] in (1:319)
			countVal[1]+=1
		elseif ClusterList[i].pts[j] in (320:715)
			countVal[2]+=1
		elseif ClusterList[i].pts[j] in (716:1112)
			countVal[3]+=1
		elseif ClusterList[i].pts[j] in (1113:1508)
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
println("Avg Purity: ",AvgPurity/1508)
