import dspy


########################################################################
########################################################################
##################  SIGNATURES FOR CONTEXT GATHERER  ###################
########################################################################
########################################################################



class IdentifyMissingInfoSignature(dspy.Signature):
    """
    You are an AI that reads an SMT requirement document and identifies placeholders or missing references.
    Return them as a list of queries/definitions you want to fetch externally.
    """
    document = dspy.InputField(prefix="Here is the requirement doc:\n", format=str)
    queries = dspy.OutputField(
        prefix="List all missing info or unclear references. One per line:\n",
        format=str
    )


class DraftSmTSignature(dspy.Signature):
    """
    You are an AI that generates SMT-LIB code from:
    - The original requirement doc
    - Additional context or definitions that were retrieved
    """
    requirement_doc = dspy.InputField(prefix="Requirement doc:\n", format=str)
    context_info = dspy.InputField(prefix="Context info:\n", format=str)
    smt_code = dspy.OutputField(prefix="Output the SMT code only:\n", format=str)


class RefineSmTSignature(dspy.Signature):
    """
    You are an AI that refines SMT code if it's unsatisfiable or has issues.
    Keep the structure as much as possible, only fix or clarify constraints.
    """
    smt_code = dspy.InputField(prefix="SMT code:\n", format=str)
    reason = dspy.InputField(prefix="Reason for refinement:\n", format=str)
    refined_smt_code = dspy.OutputField(
        prefix="Output updated (fixed) SMT code:\n", format=str
    )



########################################################################
########################################################################
######  SIGNATURES FOR SMT PROGRAMMER -- REQUIREMENT SPLITTER  ######################
########################################################################
########################################################################


class RemoveDuplicatesSignature(dspy.Signature):
    """
    Removes duplicates from enumerated statements using an LLM.
    """
    prompt = dspy.InputField(
        prefix="Below is a list of statements with potential duplicates.\n",
        format=str
    )
    llm_output = dspy.OutputField(
        prefix="Provide the revised statements or indices to remove.\n",
        format=str
    )


class ImproveAtomicitySignature(dspy.Signature):
    """
    Improves the atomicity of a statement using an LLM.
    """
    prompt = dspy.InputField(
        prefix="Rewrite the following statement to be more atomic:\n",
        format=str
    )
    llm_output = dspy.OutputField(
        prefix="Here is the more atomic version:\n",
        format=str
    )


class ExtractStatementsSignature(dspy.Signature):
    """
    Extracts statements from raw text via LLM.
    """
    prompt = dspy.InputField(
        prefix="Extract statements from the following text:\n",
        format=str
    )
    llm_output = dspy.OutputField(
        prefix="Here is the list of statements:\n",
        format=str
    )


class ImproveSelfContainednessSignature(dspy.Signature):
    """
    Rewrite a statement to be more self-contained via LLM.
    """
    prompt = dspy.InputField(
        prefix="Rewrite the following statement so it is self-contained:\n",
        format=str
    )
    llm_output = dspy.OutputField(
        prefix="Here is the self-contained statement:\n",
        format=str
    )


########################################################################
########################################################################
#######  SIGNATURES FOR SMT PROGRAMMER -- REQUIREMENT TRANSLATOR #######
########################################################################
########################################################################


class ConvertToSMTSignature(dspy.Signature):
    """
    translates single statement into the corresponding logical form
    """

    prompt = dspy.InputField(
        prefix="Translate the following statement into SMT formula",
        format=str
    )

    llm_output = dspy.OutputField(
        prefix="Here is the translated SMT formula",
        format=str
    )


class StatementVerifierignature(dspy.Signature):
    """
    Verify the translated statement and do correction
    """

    prompt = dspy.InputField(
        prefix="Verify the following SMT statement based on the corresponding text description and the context",
        format=str
    )

    llm_output = dspy.OutputField(
        prefix="Here is the corrected SMT statement",
        format=str
    )



