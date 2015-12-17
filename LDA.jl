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
		else 
			if file[1]!='.'
				println("Pushing ",file," into docs")
				push!(docs,abspath(file))
			end
		end
	end
end


#-----------------------------#
#Function to read and pre-process data



function prepro()

f=open("../DataSet/StopWords/common-english-words.txt")
stopWords=split(readall(f),',')
append!(stopWords,["from","subject","organization","distribution","lines","re","","com","edu","don't","writes","article","i'm"])

println("Processed Stop Words")

newWords={}
new={}
lines={}
line_punct=['\n']
punctuations=['.',',',':',';','?','!','<','>','(',')','{','}','-',' ','\n','\r','"','_','/','*']
for i=1:noOfDocs
	println("Processing Doc $i")
	f=open(docs[i])
	s=lowercase(readall(f))
	lines=split(s,line_punct)
	words_in_doc={}
	for j=1:length(lines)
		p = match(r"^(.*)(:)(.*)$", lines[j])
		if p!= nothing
			continue
		end
		newWords=split(lines[j],punctuations)
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
		append!(words_in_doc,new)
		splice!(new,1:length(new))
	end
	word_doc[i]=words_in_doc
	append!(all_words,words_in_doc)
	close(f)
	println("Processed words of document $i")
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
		println("Initial assignment over for topic $i")
	end
end

#-----------------------------#
#Function for initial Random assignment

function rand_assign()
	for i=1:noOfDocs
		println("Working on document: ",i,"\n")
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
	end
end

#-----------------------------#
#Function to perform Collapsed Gibbs Sampling

function gibbss()
	flag=1
	for iter=1:10						#Do for 100 iterations
		z_val=1
		GS_val=zeros(Float64,k)
		flag=0
		for i=1:noOfDocs
			println("Iter $iter Working on document: ",i,"\n")
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

docs={}
start_dir=pwd()
current_dir="../DataSet/20Newsgroups"

cd(current_dir)
goInto(current_dir)
cd(start_dir)

noOfDocs=length(docs)
println("Number of documents in total: ",noOfDocs)

all_words={}
word_doc=Array(Vector{String}, noOfDocs)

prepro()

un=unique(all_words)

#-----------------------------#
#Print all and unique words in corpus and document-word matrix

#println("ALL WORDS:\n",all_words)
#println("UNIQUES WORDS:\n",un)
#println("DOC-WORD:\n",word_doc)

#-----------------------------#
#Variable declaration for LDA

alpha=0.1
beta=0.01

k=4

type z
	word::String
	topic::Int64
end
type distribution
	item
	freq::Float64
end

z_assign= Array(z,0)
word_dist=Array(Vector{distribution},k)
topic_dist=Array(Vector{distribution},noOfDocs)
tot_in_top=zeros(Int64,k)

#-----------------------------#
#Random initial assignment

init_word()
rand_assign()

#-----------------------------#
#Print the assignment, word and topic distributions

#println("ASSIGNMENT:\n",z_assign)
#println("WORD DISTRIBUTION:\n",word_dist)
#println("TOPIC DISTRIBUTION:\n",topic_dist)


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
#Sort the word distribution over the topics

freqArr=Array(Int64,length(un))						
wordDistNew=Array(Vector{distribution},k)
for i=1:k
	wordDistNew[i]={}									
end
for i=1:k											
	for j=1:length(un)
		freqArr[j]=word_dist[i][j].freq
	end
	sortedInd=sortperm(freqArr,rev=true)
	#println(sortedInd)
	
	for j=1:length(un)
		push!(wordDistNew[i],distribution("",0))
		wordDistNew[i][j]=word_dist[i][sortedInd[j]]
	end	
end

#-----------------------------#
#Print top 10 words in each topic

top_words=Array(Vector{String},k)
for i=1:k
	top_words[i]={}
end
for i=1:k
	for j=1:10
		push!(top_words[i], wordDistNew[i][j].item)
	end
end
for i=1:k
	println("Topic ",i,":\n",top_words[i])
end
