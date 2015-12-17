#-----------------------------#
#Function to traverse through folder structure

function goInto(dir)
	for file in readdir(dir)
		if isdir(file)
			#println(file," is a directory.\n")
			cd(file)
			#println("Going into ",dir,"->",file,"\n")
			goInto(dir*"/"*file)
			cd(dir)
		elseif !contains(file,".")
			if file[1]!='.'
				println("Pushing ",file," into docs")
				push!(docs,abspath(file))
			end
		end
	end
end

#-----------------------------#
#Function to read and pre-process data in document

@everywhere function doc_process(lines)
	f=open("../DataSet/StopWords/common-english-words.txt")
	stopWords=split(readall(f),',')
	append!(stopWords,[""])
	newWords={}
	new={}
	punctuations=['.',',',':',';','?','!','<','>','(',')','{','}','-',' ','\r','"','_','/','*','\t']
	words_in_doc={}
	for j=1:length(lines)
		p = match(r"^(.*)(:)(.*)$", lines[j])			#To remove to, from and other such tags
		if p!= nothing
			continue
		end
		newWords=split(lines[j],punctuations)
		#println("Split into words for line ",j,".\n")
		append!(new,newWords)
		for l=1:length(new)
		p = match(r"^\s*(.*)\s*(.*)$", new[l])
		new[l]=p.captures[1]
		append!(new,[convert(ASCIIString,p.captures[2])])
	end
	for l=1:length(new)
		p = match(r"^[^A-Za-z0-9]?(.*)[^A-Za-z0-9]?$", new[l])
		new[l]=p.captures[1]
	end
	for l=1:length(new)
		p = match(r"^[A-Za-z]*(-|')?[A-Za-z]*$", new[l])
		if p== nothing
			new[l]=""
		end
	end

		delIndices=findin(new,stopWords)
		deleteat!(new,delIndices)
		#println("Removed stop words.\n")
		append!(words_in_doc,new)
		splice!(new,1:length(new))
	end
	return words_in_doc
end

#-----------------------------#
#Function to read and pre-process data in corpus

function prepro()
	lines={}
	line_punct=['\n']
	for i=1:noOfDocs
		f=open(docs[i])
		println("Opened document ",docs[i],"\n",i,"\n")
		s=lowercase(readall(f))
		lines=split(s,line_punct)
		#println("Split into lines.\nLength: ",length(lines),"\n")
		words_in_doc={}
		words_in_doc=fetch(@spawn doc_process(lines))
		#println("Processed document ",i,"\n")
		word_doc[i]=words_in_doc
		#println(word_doc[i])
		append!(all_words,words_in_doc)
		close(f)
	end
end

#-----------------------------#
#Function to initialize word_dist matrix

function init_word()
	for i=1:k
		word_dist[i]={}
		for j=1:length(un)
			push!(word_dist[i],distribution(un[j],0))						#Store unique words and 0 frequency
		end
		#println("WORDDIST",i,"\n",word_dist[i])
	end
end

#-----------------------------#
#Function for initial Random assignment for a document

@everywhere function doc_assign_rand(word_doc,un,tot_in_top,z_assign,word_dist,topic_dist,i)
	list_topics={}
	for j=1:length(word_doc[i])
		pos=0
		topic_ass=rand(1:k)						#Pick random topic
												#Modify data structures accordingly
		tot_in_top[topic_ass]+=1
		push!(list_topics,topic_ass)
		push!(z_assign,z(word_doc[i][j],topic_ass))
		index=findin(un,[word_doc[i][j]])
		pos=index[1]
		word_dist[topic_ass][pos].freq+=1
	end
	topic_dist[i]={}
	for l=1:k
		ind=findin(list_topics,l)
		push!(topic_dist[i], distribution(l,length(ind)))
	end
	return tot_in_top,z_assign,word_dist,topic_dist
end

#-----------------------------#
#Function to perform Collapsed Gibbs Sampling

function gibbss()
	flag=1
	for iter=1:100						#Do for 100 iterations
		z_val=1
		GS_val=zeros(Float64,k)
		flag=0
		for i=1:noOfDocs
			#println("Working on document: ",i,"\n")
			for j=1:length(word_doc[i])
				#println("\tWorking on word: ",j,"\n")
				top_assigned=z_assign[z_val].topic						#Obtain currently assigned topic
				pos=findin(un,[z_assign[z_val].word])
																		#Remove assignment from all structures
				word_dist[top_assigned][pos[1]].freq-=1
				topic_dist[i][top_assigned].freq-=1
				tot_in_top[top_assigned]-=1
				for l=1:k
																	#Calculate Gibbs Sampling value for all topics
					#println("\t\tCalculating for topic: ",l,"\n")
					val1=word_dist[l][pos[1]].freq
					val2=topic_dist[i][l].freq
					prob1=(val1+beta)/(tot_in_top[l]+(beta*length(word_dist[l])))
					prob2=(val2+alpha)
					GS_val[l]=prob1*prob2
				end
				for l=2:k
					GS_val[l]+=GS_val[l-1]								#Cumulating the probabilities, to pick new topic
				end
																		#Pick new topic
				pick=rand()*GS_val[k]
				new_top=1
				for new_top=1:k
					if pick<GS_val[new_top]
						break
					end
				end
																		#Modify structures--Add to new topic
				z_assign[z_val].topic=new_top
				word_dist[new_top][pos[1]].freq+=1
				topic_dist[i][new_top].freq+=1
				tot_in_top[new_top]+=1
				z_val+=1
				if top_assigned!=new_top
					flag+=1
				end
			end
		end
		println("Reassignments in iteration : ",flag)
	end
end

#-----------------------------#
#LDA MAIN FUNCTIONALITY
#-----------------------------#

#-----------------------------#
#Read and pre-process the corpus


@everywhere docs={}
start_dir=pwd()
current_dir="../DataSet/20Newsgroups"

cd(current_dir)
goInto(current_dir)
cd(start_dir)

@everywhere noOfDocs=length(docs)

println("Number of documents in total: ",noOfDocs)

@everywhere all_words={}
@everywhere word_doc=Array(Vector{String}, noOfDocs)

prepro()

#=
@everywhere un=unique(all_words)

#-----------------------------#
#Print all and unique words in corpus and document-word matrix

#println("ALL WORDS:\n",all_words)
#println("UNIQUES WORDS:\n",un)
#println("DOC-WORD:\n",word_doc)

#-----------------------------#
#Variable declaration for LDA

alpha=0.1
beta=0.01

@everywhere k=2

@everywhere type z
	word::String
	topic::Int64
end
@everywhere type distribution
	item
	freq::Float64
end

@everywhere z_assign= Array(z,0)
@everywhere word_dist=Array(Vector{distribution},k)
@everywhere topic_dist=Array(Vector{distribution},noOfDocs)
@everywhere tot_in_top=zeros(Int64,k)

#-----------------------------#
#Random initial assignment

init_word()
for i=1:noOfDocs
		println("Working on document: ",i,"\n")
		tot_in_top,z_assign,word_dist,topic_dist=fetch(@spawn doc_assign_rand(word_doc,un,tot_in_top,z_assign,word_dist,topic_dist,i))
end


#-----------------------------#
#Print the assignment, word and topic distributions

println("ASSIGNMENT:\n",z_assign)
println("WORD DISTRIBUTION:\n",word_dist)
println("TOPIC DISTRIBUTION:\n",topic_dist)

#-----------------------------#
#Gibbs Sampling
gibbss()

#-----------------------------#
#Print word and topic distributions

for i=1:k
	println("WORD DISTRIBUTION ",i,":\n",word_dist[i])
end

for i=1:noOfDocs
	println("TOPIC DISTRIBUTION ",i,":\n",topic_dist[i])
end

#-----------------------------#
#Print top 10 words in each topic

top_words=Array(Vector{String},k)
for i=1:k
	top_words[i]={}
end
for iter=1:10
	for i=1:k
		max=0
		pos=0
		for j=1:length(word_dist[i])
			if word_dist[i][j].freq>max && in(word_dist[i][j].item,top_words[i])==false
				max=word_dist[i][j].freq
				pos=j
			end
		end
		push!(top_words[i],word_dist[i][pos].item)
	end
end

for i=1:k
	println("Topic ",i,":\n",top_words[i])
end
=#
