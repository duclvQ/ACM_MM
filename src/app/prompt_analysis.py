from langchain_community.llms import Ollama


template = """
## Meeting Minutes

**Date:** [Date]
**Time:** [Start Time] - [End Time]
**Location:** [Location]
**Attendees:**
- [Person 1]
- [Person 2]
- [Person 3]

### Agenda
1. [Agenda Item 1]
2. [Agenda Item 2]
3. [Agenda Item 3]

### Discussion Summary
- **[Agenda Item 1]**: [Discussion Points]
- **[Agenda Item 2]**: [Discussion Points]
- **[Agenda Item 3]**: [Discussion Points]

### Action Items
- [Action Item 1]: [Person Responsible] - [Deadline]
- [Action Item 2]: [Person Responsible] - [Deadline]
- [Action Item 3]: [Person Responsible] - [Deadline]

### Decisions Made
- [Decision 1]
- [Decision 2]
- [Decision 3]

**Next Meeting:**
- **Date:** [Next Meeting Date]
- **Time:** [Next Meeting Time]
- **Location:** [Next Meeting Location]

**Minutes Prepared by:**
[Your Name]
"""



class MeetingSummaryEvaluator:
    def __init__(self, llm_model, meeting_details, template = template, max_iterations=5):
        self.llm = Ollama(model=llm_model)
        self.meeting_details = meeting_details
        self.template = template
        self.max_iterations = max_iterations

    def generate_summary(self, prompt):
        prompt = f"""{prompt}"""
        result = self.llm.generate([prompt])
        summary = result.generations[0][0].text
        return summary

    def generate_questions(self):
        prompt = f"""
        Based on the following meeting details, generate questions to evaluate the accuracy of the summary. The questions should be based on a meeting minutes template and should be in one of the following formats: yes/no questions, short answers, names only, or numbers only.
        I need 10 questions.
        Do not generate answers, generate questions only.
        
        Meeting Details:
        {self.meeting_details}

        Questions:
        """
        result = self.llm.generate([prompt])
        questions = result.generations[0][0].text.split("\n")
        return questions

    def extract_answers(self, text, questions):
        answers = []
        for question in questions:
            prompt = f"""
            Based on the following summary of the meeting minutes, answer the question briefly. Use only yes/no, short answers, names, or numbers. 
            Do not provide explanations.
            Do not explain the answer.

            Summary:
            {text}

            Question:
            {question}

            Answer:
            """
            result = self.llm.generate([prompt])
            answer = result.generations[0][0].text.strip()
            answers.append(answer)
        return answers

    def compare_answers(self, source_answers, summary_answers):
        consistency_score = 0
        for source, summary in zip(source_answers, summary_answers):
            if source.lower() == summary.lower():
                consistency_score += 1
        return round(consistency_score / len(source_answers), 2)

    def evaluate_summary_factual_consistency(self):
        questions = self.generate_questions()

        best_consistency_score = 0
        best_summary = None
        best_source_answers = None
        best_summary_answers = None

        iterations = 0

        while best_consistency_score < 0.6 and iterations < self.max_iterations:
            summary = self.generate_summary(self.get_prompt())
            source_answers = self.extract_answers(self.meeting_details, questions)
            summary_answers = self.extract_answers(summary, questions)
            consistency_score = self.compare_answers(source_answers, summary_answers)

            print(f"Iteration {iterations + 1}, Consistency Score: {consistency_score}")

            if consistency_score > best_consistency_score:
                best_consistency_score = consistency_score
                best_summary = summary
                best_source_answers = source_answers
                best_summary_answers = summary_answers

            if best_consistency_score < 0.6:
                prompt = self.get_prompt(questions)

            iterations += 1

        return {
            "summary": best_summary,
            "questions": questions,
            "source_answers": best_source_answers,
            "summary_answers": best_summary_answers,
            "consistency_score": best_consistency_score,
            "iterations": iterations
        }

    def get_prompt(self, questions=None):
        if questions is None:
            return f"""
            I’m Meeting summary officer, please help me to summarize using the meeting minute template.
            We have finished the meeting:

            {self.meeting_details}

            Use this template:

            {self.template}

            Do not enter information that is not displayed.
            """
        else:
            return f"""
            I’m Meeting summary officer, please help me to summarize using the meeting minute template.
            We have finished the meeting:

            {self.meeting_details}

            Use this template:

            {self.template}

            Do not enter information that is not displayed.

            Make sure the summary can accurately answer these questions:

            {questions}
            """