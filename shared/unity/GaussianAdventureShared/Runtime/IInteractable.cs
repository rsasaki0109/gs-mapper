namespace NerfGsPlayground.GaussianAdventureShared
{
    /// <summary>
    /// DreamWalker の簡易 interaction 契約。
    /// まずは画面中央 raycast による一人称探索を前提にしている。
    /// </summary>
    public interface IInteractable
    {
        bool CanInteract(SplatInteractProbe probe);
        string GetInteractionPrompt(SplatInteractProbe probe);
        void Interact(SplatInteractProbe probe);
    }
}
