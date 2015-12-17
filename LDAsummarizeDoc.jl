println("Document Summary\n\nEnter document number: ")
input=int(chomp(readline(STDIN)))
sum=0
for i=1:k
	sum+=topic_dist[input][i].freq
end
#summary=Array(String, 11)
summary={}
for i=1:k
	num=int((topic_dist[input][i].freq/sum)*10)
	wordsInTopic=Array(String, length(un))
	for j=1:length(un)
		wordsInTopic[j]=wordDistNew[i][j].item
	end
	presentInd=findin(wordsInTopic,word_doc[input])
	for j=1:num
		push!(summary,wordsInTopic[presentInd[j]])
	end
end

println("Summary: ")
println(summary)
